"""Instrument convolution + binning (v1-D JAX port).

Mirrors `POSEIDON/POSEIDON/instrument.py:321-396` (`make_model_data`)
and the multi-dataset wrapper at `:399-447` (`bin_spectrum_to_data`).

Supports both spectroscopic and photometric instruments. The Gaussian
PSF convolution uses ``_jax_filters.gaussian_filter1d_edge``; the
trapezoidal integration uses ``jnp.trapezoid``. Per-bin metadata is
prepared via ``_instrument_setup.compute_instrument_indices(...)``.

When called from inside ``jax.jit``, ``sigma``, ``bin_left``, ``bin_cent``,
``bin_right`` and ``norm`` must be Python sequences / numpy arrays of
static values (the kernel sizes and gather indices are traced as static).
``bin_spectrum_to_data`` therefore exposes the same ``data_properties``
dict-of-numpy-arrays interface as the v0.5 numpy version; only
``spectrum`` is a JAX tracer at retrieval time.
"""

import jax
import jax.numpy as jnp
import numpy as np

from jaxposeidon._instrument_setup import (
    PHOTOMETRIC_INSTRUMENTS,
    compute_instrument_indices,  # noqa: F401
)
from jaxposeidon._jax_filters import gaussian_filter1d_edge

jax.config.update("jax_enable_x64", True)


def make_model_data(
    spectrum,
    wl,
    sigma,
    sensitivity,
    bin_left,
    bin_cent,
    bin_right,
    norm,
    photometric=False,
):
    """Bin a fine-grid spectrum to data wavelengths via PSF convolution +
    sensitivity-weighted integration.

    Bit-exact port of POSEIDON `instrument.py:321-396`. The setup-time
    metadata (``sigma``, ``bin_left``, ``bin_right``, ``sensitivity``,
    ``norm``) is consumed as numpy at trace time; only ``spectrum`` is
    a JAX tracer under ``jit``.
    """
    spectrum = jnp.asarray(spectrum, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    sensitivity = jnp.asarray(sensitivity, dtype=jnp.float64)

    if photometric:
        integrand = (
            spectrum[bin_left[0] : bin_right[0]]
            * sensitivity[bin_left[0] : bin_right[0]]
        )
        data = jnp.trapezoid(integrand, wl[bin_left[0] : bin_right[0]])
        return jnp.atleast_1d(data / norm)

    sigma = np.asarray(sigma)
    bin_left = np.asarray(bin_left)
    bin_cent = np.asarray(bin_cent)
    bin_right = np.asarray(bin_right)
    norm = np.asarray(norm)

    N_bins = len(bin_cent)
    ymodel_list = []
    for n in range(N_bins):
        extension = int(max(1, int(2 * sigma[n])))
        slice_lo = int(bin_left[n] - extension)
        slice_hi = int(bin_right[n] + extension)
        spectrum_conv = gaussian_filter1d_edge(
            spectrum[slice_lo:slice_hi], sigma=float(sigma[n])
        )
        conv_trim = spectrum_conv[extension : spectrum_conv.shape[0] - extension]
        if conv_trim.shape[0] != int(bin_right[n] - bin_left[n]):
            raise Exception(
                "Error: Model wavelength range not wide enough to encompass all data."
            )
        integrand = conv_trim * sensitivity[int(bin_left[n]) : int(bin_right[n])]
        bin_val = jnp.trapezoid(integrand, wl[int(bin_left[n]) : int(bin_right[n])])
        ymodel_list.append(bin_val / float(norm[n]))
    return jnp.stack(ymodel_list)


def bin_spectrum_to_data(spectrum, wl, data_properties):
    """Multi-instrument spectroscopic wrapper around `make_model_data(...)`.

    Mirrors POSEIDON `instrument.py:399-447`. ``data_properties`` is the
    setup-time numpy dict from ``load_data(...)``; ``spectrum`` (and ``wl``)
    may be JAX tracers.
    """
    # wl shape is treated as static; under jit pass wl with known shape.
    N_wl = int(wl.shape[0])
    pieces = []
    instruments = data_properties["instruments"]
    len_data_idx = np.asarray(data_properties["len_data_idx"])
    for i in range(len(data_properties["datasets"])):
        instrument = instruments[i]
        idx_1 = int(len_data_idx[i])
        idx_2 = int(len_data_idx[i + 1])
        ymodel_i = make_model_data(
            spectrum,
            wl,
            data_properties["psf_sigma"][idx_1:idx_2],
            data_properties["sens"][i * N_wl : (i + 1) * N_wl],
            data_properties["bin_left"][idx_1:idx_2],
            data_properties["bin_cent"][idx_1:idx_2],
            data_properties["bin_right"][idx_1:idx_2],
            data_properties["norm"][idx_1:idx_2],
            photometric=(instrument in PHOTOMETRIC_INSTRUMENTS),
        )
        pieces.append(jnp.atleast_1d(ymodel_i))
    return jnp.concatenate(pieces)
