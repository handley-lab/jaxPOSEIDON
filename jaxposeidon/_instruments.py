"""Instrument convolution + binning — v0 port.

Mirrors `POSEIDON/POSEIDON/instrument.py:321-396` (`make_model_data`) and
the multi-dataset wrapper at `:399-447` (`bin_spectrum_to_data`).

v0 envelope: spectroscopic instruments only (photometric branch
deferred). The user prepares `(sigma, sensitivity, bin_left, bin_cent,
bin_right, norm)` via POSEIDON's `init_instrument` (which depends on
POSEIDON's reference_data instrument sensitivity files) and passes
them to `make_model_data` here.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz


def compute_instrument_indices(wl, wl_data, half_width, sensitivity, fwhm_um):
    """v0 spectroscopic equivalent of POSEIDON `instrument.py:191-318`.

    Given the model wavelength grid `wl`, the data bin centres `wl_data`
    and half-widths, the precomputed instrument sensitivity on the model
    grid, and the PSF FWHM array (μm), compute the per-bin
    `(sigma, bin_left, bin_cent, bin_right, norm)` arrays expected by
    `make_model_data(...)`.

    This is the portion of POSEIDON's `init_instrument` that does NOT
    require POSEIDON's `reference_data` sensitivity files (those are
    POSEIDON-package data; the caller can obtain them via POSEIDON's
    own `init_instrument` and pass `sensitivity` here, or supply a
    user-provided one).

    Photometric instruments are deferred to v1.
    """
    N_bins = len(wl_data)
    sigma_um = 0.424661 * np.asarray(fwhm_um)  # POSEIDON's PSF σ
    bin_left = np.zeros(N_bins, dtype=np.int64)
    bin_cent = np.zeros(N_bins, dtype=np.int64)
    bin_right = np.zeros(N_bins, dtype=np.int64)
    sigma = np.zeros(N_bins)
    norm = np.zeros(N_bins)
    for n in range(N_bins):
        bin_left[n] = int(np.argmin(np.abs(wl - (wl_data[n] - half_width[n]))))
        bin_cent[n] = int(np.argmin(np.abs(wl - wl_data[n])))
        bin_right[n] = int(np.argmin(np.abs(wl - (wl_data[n] + half_width[n]))))
        dwl = 0.5 * (wl[bin_cent[n] + 1] - wl[bin_cent[n] - 1])
        sigma[n] = sigma_um[n] / dwl
        norm[n] = trapz(
            sensitivity[bin_left[n]:bin_right[n]],
            wl[bin_left[n]:bin_right[n]],
        )
    return sigma, bin_left, bin_cent, bin_right, norm


def make_model_data(spectrum, wl, sigma, sensitivity, bin_left, bin_cent,
                    bin_right, norm, photometric=False):
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
            sensitivity[bin_left[n]:bin_right[n]]
        ):
            raise Exception(
                "Error: Model wavelength range not wide enough to encompass all data."
            )
        integrand = (
            spectrum_conv[extension:-extension]
            * sensitivity[bin_left[n]:bin_right[n]]
        )
        data[n] = trapz(integrand, wl[bin_left[n]:bin_right[n]])
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
            spectrum, wl,
            data_properties["psf_sigma"][idx_1:idx_2],
            data_properties["sens"][i * N_wl:(i + 1) * N_wl],
            data_properties["bin_left"][idx_1:idx_2],
            data_properties["bin_cent"][idx_1:idx_2],
            data_properties["bin_right"][idx_1:idx_2],
            data_properties["norm"][idx_1:idx_2],
            photometric=False,
        )
        ymodel = np.concatenate([ymodel, ymodel_i])
    return ymodel
