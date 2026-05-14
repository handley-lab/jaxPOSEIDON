"""Instrument setup-only module — extracted from `_instruments.py`.

Setup-only: numpy / scipy / file I/O permitted. Must not be called
from inside `jit`. Allow-listed by the v1 source-grep gate (see
`CLAUDE.md`).

Contents:
- `PHOTOMETRIC_INSTRUMENTS`: immutable POSEIDON-mirror photometric
  instrument list (`instrument.py:249-252`).
- `compute_instrument_indices(...)`: spectroscopic per-bin metadata.
- `compute_photometric_indices(...)`: photometric per-band metadata.
"""

import numpy as np

try:
    from numpy import trapezoid as trapz
except ImportError:
    from numpy import trapz


PHOTOMETRIC_INSTRUMENTS = frozenset({"IRAC1", "IRAC2"})


def compute_instrument_indices(wl, wl_data, half_width, sensitivity, fwhm_um):
    """v0 spectroscopic equivalent of POSEIDON `instrument.py:191-318`.

    Given the model wavelength grid `wl`, the data bin centres
    `wl_data` and half-widths, the precomputed instrument sensitivity
    on the model grid, and the PSF FWHM array (μm), compute the
    per-bin `(sigma, bin_left, bin_cent, bin_right, norm)` arrays
    expected by `_instruments.make_model_data(...)`.

    This is the portion of POSEIDON's `init_instrument` that does NOT
    require POSEIDON's `reference_data` sensitivity files (those are
    POSEIDON-package data; the caller can obtain them via POSEIDON's
    own `init_instrument`, or jaxPOSEIDON's vendored / user-pointed
    instrument-sensitivity dispatch, and pass `sensitivity` here).

    Photometric instruments are deferred to Phase 0.5.5.
    """
    N_bins = len(wl_data)
    sigma_um = 0.424661 * np.asarray(fwhm_um)
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
            sensitivity[bin_left[n] : bin_right[n]],
            wl[bin_left[n] : bin_right[n]],
        )
    return sigma, bin_left, bin_cent, bin_right, norm


def compute_photometric_indices(wl, wl_data, half_width, sensitivity, fwhm_um=0.0):
    """Photometric-band per-data-point metadata (POSEIDON instrument.py:300-316).

    For photometric bands (IRAC1, IRAC2), no PSF convolution is applied,
    so `sigma` is returned as `0.424661 * fwhm_um` (POSEIDON's dummy-value
    convention — the value is unused downstream but kept for shape parity
    with the spectroscopic helper). `wl_data` and `half_width` are scalars.
    """
    bin_left = np.array(
        [int(np.argmin(np.abs(wl - (wl_data - half_width))))], dtype=np.int64
    )
    bin_cent = np.array([int(np.argmin(np.abs(wl - wl_data)))], dtype=np.int64)
    bin_right = np.array(
        [int(np.argmin(np.abs(wl - (wl_data + half_width))))], dtype=np.int64
    )
    sigma = np.array([0.424661 * float(fwhm_um)])
    norm = np.array(
        [
            trapz(
                sensitivity[bin_left[0] : bin_right[0]],
                wl[bin_left[0] : bin_right[0]],
            )
        ]
    )
    return sigma, bin_left, bin_cent, bin_right, norm
