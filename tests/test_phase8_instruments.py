"""Phase 8 tests for _instruments + _data against POSEIDON."""

import numpy as np
import pytest

from jaxposeidon import _data, _instruments


def test_make_model_data_matches_poseidon():
    from POSEIDON.instrument import make_model_data as p_mmd
    rng = np.random.default_rng(0)
    N_wl, N_bins = 500, 20
    wl = np.linspace(1.0, 5.0, N_wl)
    spectrum = rng.uniform(2e-3, 3e-3, size=N_wl)
    bin_centres_wl = np.linspace(1.5, 4.5, N_bins)
    bin_cent = np.array([int(np.argmin(np.abs(wl - c))) for c in bin_centres_wl])
    bin_left = np.maximum(bin_cent - 5, 0)
    bin_right = np.minimum(bin_cent + 5, N_wl - 1)
    sigma = np.full(N_bins, 1.5)  # PSF in grid units
    sensitivity = rng.uniform(0.1, 1.0, size=N_wl)
    norm = np.array([
        np.trapezoid(sensitivity[bin_left[i]:bin_right[i]],
                     wl[bin_left[i]:bin_right[i]])
        if hasattr(np, "trapezoid")
        else np.trapz(sensitivity[bin_left[i]:bin_right[i]],
                       wl[bin_left[i]:bin_right[i]])
        for i in range(N_bins)
    ])
    ours = _instruments.make_model_data(spectrum, wl, sigma, sensitivity,
                                         bin_left, bin_cent, bin_right, norm)
    theirs = p_mmd(spectrum, wl, sigma, sensitivity,
                   bin_left, bin_cent, bin_right, norm,
                   photometric=False)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_make_model_data_rejects_photometric():
    with pytest.raises(NotImplementedError, match="Photometric"):
        _instruments.make_model_data(
            np.zeros(10), np.linspace(0, 1, 10), np.zeros(1),
            np.ones(10), np.array([0]), np.array([1]), np.array([2]),
            np.array([1.0]), photometric=True,
        )


# ---------------------------------------------------------------------------
# Likelihood / offsets / error inflation
# ---------------------------------------------------------------------------
def test_loglikelihood_no_offsets_no_inflation():
    rng = np.random.default_rng(1)
    n = 20
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = np.full(n, 1e-4)
    ll = _data.loglikelihood(ymodel, ydata, err_data)
    expected = (-0.5 * ((ymodel - ydata) / err_data) ** 2).sum()
    np.testing.assert_allclose(ll, expected, atol=0)


def test_loglikelihood_rejects_NaN_spectrum():
    n = 5
    ydata = np.full(n, 3e-3)
    ymodel = np.array([3e-3, 3e-3, np.nan, 3e-3, 3e-3])
    err_data = np.full(n, 1e-4)
    assert _data.loglikelihood(ymodel, ydata, err_data) == -1.0e100


def test_apply_offsets_single_dataset():
    ydata = np.arange(10, dtype=float) * 1e-3
    out = _data.apply_offsets(ydata, np.array([100.0]),  # +100 ppm
                                offsets_applied="single_dataset",
                                offset_start=2, offset_end=6)
    expected = ydata.copy()
    expected[2:6] -= 100.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


def test_apply_offsets_returns_unchanged_when_None():
    ydata = np.arange(5, dtype=float)
    np.testing.assert_array_equal(
        _data.apply_offsets(ydata, np.array([]), None, 0, 0), ydata,
    )


def test_effective_error_sq_no_inflation():
    err = np.array([1.0, 2.0, 3.0])
    err_eff_sq, norm_log = _data.effective_error_sq(
        err, np.zeros_like(err), np.array([]), None, norm_log_default=0.123,
    )
    np.testing.assert_array_equal(err_eff_sq, err * err)
    assert norm_log == 0.123


def test_effective_error_sq_Line15():
    rng = np.random.default_rng(0)
    err = rng.uniform(1e-4, 1e-3, size=8)
    ymodel = rng.uniform(2.5e-3, 3e-3, size=8)
    params = np.array([-8.0])
    err_eff_sq, norm_log = _data.effective_error_sq(
        err, ymodel, params, "Line15",
    )
    expected = err * err + 10.0 ** params[0]
    np.testing.assert_array_equal(err_eff_sq, expected)
    np.testing.assert_allclose(
        norm_log, (-0.5 * np.log(2.0 * np.pi * expected)).sum(),
    )


def test_effective_error_sq_Piette20():
    rng = np.random.default_rng(1)
    err = rng.uniform(1e-4, 1e-3, size=8)
    ymodel = rng.uniform(2.5e-3, 3e-3, size=8)
    params = np.array([0.5])
    err_eff_sq, _ = _data.effective_error_sq(err, ymodel, params, "Piette20")
    expected = err * err + (params[0] * ymodel) ** 2
    np.testing.assert_array_equal(err_eff_sq, expected)


def test_effective_error_sq_combined():
    rng = np.random.default_rng(2)
    err = rng.uniform(1e-4, 1e-3, size=8)
    ymodel = rng.uniform(2.5e-3, 3e-3, size=8)
    params = np.array([-9.0, 0.3])
    err_eff_sq, _ = _data.effective_error_sq(
        err, ymodel, params, "Line15+Piette20",
    )
    expected = err * err + 10.0 ** params[0] + (params[1] * ymodel) ** 2
    np.testing.assert_array_equal(err_eff_sq, expected)
