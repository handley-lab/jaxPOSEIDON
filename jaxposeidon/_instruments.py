"""Instrument convolution + binning — v0 port.

Mirrors `POSEIDON/POSEIDON/instrument.py:321-396` (`make_model_data`)
and the multi-dataset wrapper at `:399-447` (`bin_spectrum_to_data`).

v0 envelope: spectroscopic instruments only (photometric branch
deferred to Phase 0.5.5). The user prepares
`(sigma, sensitivity, bin_left, bin_cent, bin_right, norm)` via
`jaxposeidon._instrument_setup.compute_instrument_indices(...)` and
passes them to `make_model_data` here.

Setup-only `compute_instrument_indices` lives in
`jaxposeidon._instrument_setup` (extracted Phase 0.5.2a) so this
module can be JAX-pure under the v1 source-grep gate.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz

# Back-compat re-export: existing v0 tests and external callers import
# compute_instrument_indices from jaxposeidon._instruments. Keeping the
# name here as a re-export avoids breaking the v0 1,435-test gate while
# the actual implementation lives in the setup-only module.
from jaxposeidon._instrument_setup import compute_instrument_indices  # noqa: F401


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

    Bit-exact port of POSEIDON `instrument.py:321-396`.
    """
    if photometric:
        raise NotImplementedError("Photometric instruments deferred to v1")

    N_bins = len(bin_cent)
    data = np.zeros(N_bins)
    ymodel = np.zeros(N_bins)
    for n in range(N_bins):
        extension = max(1, int(2 * sigma[n]))
        slice_lo = bin_left[n] - extension
        slice_hi = bin_right[n] + extension
        spectrum_conv = gaussian_filter1d(
            spectrum[slice_lo:slice_hi], sigma=sigma[n], mode="nearest"
        )
        if len(spectrum_conv[extension:-extension]) != len(
            sensitivity[bin_left[n] : bin_right[n]]
        ):
            raise Exception(
                "Error: Model wavelength range not wide enough to encompass all data."
            )
        integrand = (
            spectrum_conv[extension:-extension]
            * sensitivity[bin_left[n] : bin_right[n]]
        )
        data[n] = trapz(integrand, wl[bin_left[n] : bin_right[n]])
        ymodel[n] = data[n] / norm[n]
    return ymodel


def bin_spectrum_to_data(spectrum, wl, data_properties):
    """Multi-instrument spectroscopic wrapper around `make_model_data(...)`.

    Mirrors POSEIDON `instrument.py:399-447` for the spectroscopic case
    only. Photometric datasets (IRAC1, IRAC2) raise NotImplementedError.

    `data_properties` is the dict POSEIDON's `load_data(...)` returns,
    with keys:
        datasets, instruments, psf_sigma, sens, bin_left, bin_cent,
        bin_right, norm, len_data_idx.
    """
    ymodel = np.array([])
    N_wl = len(wl)
    for i in range(len(data_properties["datasets"])):
        if data_properties["instruments"][i] in ("IRAC1", "IRAC2"):
            raise NotImplementedError(
                f"Photometric instrument {data_properties['instruments'][i]} "
                "deferred to v1"
            )
        idx_1 = data_properties["len_data_idx"][i]
        idx_2 = data_properties["len_data_idx"][i + 1]
        ymodel_i = make_model_data(
            spectrum,
            wl,
            data_properties["psf_sigma"][idx_1:idx_2],
            data_properties["sens"][i * N_wl : (i + 1) * N_wl],
            data_properties["bin_left"][idx_1:idx_2],
            data_properties["bin_cent"][idx_1:idx_2],
            data_properties["bin_right"][idx_1:idx_2],
            data_properties["norm"][idx_1:idx_2],
            photometric=False,
        )
        ymodel = np.concatenate([ymodel, ymodel_i])
    return ymodel
