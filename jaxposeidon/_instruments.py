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
from numpy import trapz


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
