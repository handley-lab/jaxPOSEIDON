"""Phase 0.5.5: photometric-instrument support.

Ports POSEIDON `instrument.py:249-252, 300-316, 387-394`:
- PHOTOMETRIC_INSTRUMENTS dispatch table (IRAC1, IRAC2)
- compute_photometric_indices: per-band metadata (sigma, bin_left,
  bin_cent, bin_right, norm)
- make_model_data(photometric=True): sensitivity-weighted band integral
  with no PSF convolution

Parity tests against POSEIDON's init_instrument photometric branch and
make_model_data photometric branch.
"""

import numpy as np
import pytest

from jaxposeidon import _instruments
from jaxposeidon._instrument_setup import (
    PHOTOMETRIC_INSTRUMENTS,
    compute_photometric_indices,
)


def test_photometric_instruments_table():
    """POSEIDON instrument.py:249-252 flags IRAC1/IRAC2 only."""
    assert PHOTOMETRIC_INSTRUMENTS == {"IRAC1", "IRAC2"}


def test_compute_photometric_indices_returns_singleton_arrays():
    wl = np.linspace(3.0, 5.0, 200)
    sensitivity = np.zeros(200)
    sensitivity[50:150] = 1.0
    sigma, bin_left, bin_cent, bin_right, norm = compute_photometric_indices(
        wl,
        wl_data=4.0,
        half_width=0.5,
        sensitivity=sensitivity,
        fwhm_um=0.0,
    )
    assert (
        sigma.shape
        == bin_left.shape
        == bin_cent.shape
        == bin_right.shape
        == norm.shape
        == (1,)
    )
    assert sigma[0] == 0.0
    assert bin_left[0] < bin_cent[0] < bin_right[0]


@pytest.mark.parametrize("instrument", ["IRAC1", "IRAC2"])
def test_compute_photometric_indices_matches_poseidon(instrument):
    """Direct parity against POSEIDON init_instrument photometric branch."""
    from POSEIDON.instrument import init_instrument as p_init

    wl = np.linspace(1.0, 10.0, 5000)
    # IRAC1 ~ 3.6μm, IRAC2 ~ 4.5μm — POSEIDON's reference files determine
    # the exact sensitivity, so we let POSEIDON compute the metadata then
    # build the same sensitivity / fwhm inputs for our helper.
    wl_data = np.array([3.6]) if instrument == "IRAC1" else np.array([4.5])
    half_width = np.array([0.3])
    p_sigma, p_fwhm, p_sens, p_bl, p_bc, p_br, p_norm = p_init(
        wl, wl_data, half_width, instrument
    )
    ours = compute_photometric_indices(
        wl,
        wl_data[0],
        half_width[0],
        p_sens,
        fwhm_um=p_fwhm[0],
    )
    np.testing.assert_array_equal(ours[0], p_sigma)
    np.testing.assert_array_equal(ours[1], p_bl)
    np.testing.assert_array_equal(ours[2], p_bc)
    np.testing.assert_array_equal(ours[3], p_br)
    np.testing.assert_allclose(ours[4], p_norm, atol=0, rtol=1e-13)


@pytest.mark.parametrize("instrument", ["IRAC1", "IRAC2"])
def test_make_model_data_photometric_matches_poseidon(instrument):
    """End-to-end photometric binning parity against POSEIDON.make_model_data."""
    from POSEIDON.instrument import (
        init_instrument as p_init,
        make_model_data as p_make,
    )

    rng = np.random.default_rng(42)
    wl = np.linspace(1.0, 10.0, 5000)
    spectrum = 0.01 + 0.001 * rng.standard_normal(5000)
    wl_data = np.array([3.6]) if instrument == "IRAC1" else np.array([4.5])
    half_width = np.array([0.3])
    sigma, fwhm, sens, bl, bc, br, norm = p_init(wl, wl_data, half_width, instrument)
    ours = _instruments.make_model_data(
        spectrum,
        wl,
        sigma,
        sens,
        bl,
        bc,
        br,
        norm,
        photometric=True,
    )
    theirs = p_make(spectrum, wl, sigma, sens, bl, bc, br, norm, photometric=True)
    np.testing.assert_allclose(ours, np.atleast_1d(theirs), atol=0, rtol=1e-13)


def test_bin_spectrum_to_data_photometric_routes_via_PHOTOMETRIC_table():
    """IRAC1 in data_properties['instruments'] activates the photometric branch."""
    from POSEIDON.instrument import init_instrument as p_init

    wl = np.linspace(1.0, 10.0, 5000)
    spectrum = 0.01 * np.ones(5000)
    wl_data = np.array([3.6])
    half_width = np.array([0.3])
    sigma, fwhm, sens, bl, bc, br, norm = p_init(wl, wl_data, half_width, "IRAC1")
    data_properties = {
        "datasets": ["IRAC1"],
        "instruments": ["IRAC1"],
        "psf_sigma": sigma,
        "sens": sens,
        "bin_left": bl,
        "bin_cent": bc,
        "bin_right": br,
        "norm": norm,
        "len_data_idx": np.array([0, 1]),
    }
    result = _instruments.bin_spectrum_to_data(spectrum, wl, data_properties)
    assert result.shape == (1,)
    assert np.isfinite(result[0])
    # For a flat spectrum 0.01, the sensitivity-weighted average should be 0.01.
    np.testing.assert_allclose(result, np.array([0.01]), atol=1e-15, rtol=1e-12)
