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
def test_loglikelihood_no_offsets_no_inflation_unnormalised():
    """Caller-supplied norm_log_default=0.0 yields chi² only."""
    rng = np.random.default_rng(1)
    n = 20
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = np.full(n, 1e-4)
    ll = _data.loglikelihood(ymodel, ydata, err_data, norm_log_default=0.0)
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


# ---------------------------------------------------------------------------
# bin_spectrum_to_data multi-dataset wrapper
# ---------------------------------------------------------------------------
def test_bin_spectrum_to_data_matches_poseidon():
    from POSEIDON.instrument import bin_spectrum_to_data as p_bs
    rng = np.random.default_rng(7)
    N_wl, N_bins_a, N_bins_b = 200, 8, 10
    wl = np.linspace(1.0, 5.0, N_wl)
    spectrum = rng.uniform(2e-3, 3e-3, size=N_wl)
    # Per-dataset bin info
    cent_a = np.linspace(1.5, 2.5, N_bins_a)
    cent_b = np.linspace(3.0, 4.5, N_bins_b)
    bl_a = np.array([int(np.argmin(np.abs(wl - (c - 0.05)))) for c in cent_a])
    bc_a = np.array([int(np.argmin(np.abs(wl - c))) for c in cent_a])
    br_a = np.array([int(np.argmin(np.abs(wl - (c + 0.05)))) for c in cent_a])
    bl_b = np.array([int(np.argmin(np.abs(wl - (c - 0.05)))) for c in cent_b])
    bc_b = np.array([int(np.argmin(np.abs(wl - c))) for c in cent_b])
    br_b = np.array([int(np.argmin(np.abs(wl - (c + 0.05)))) for c in cent_b])
    sens_a = rng.uniform(0.1, 1.0, size=N_wl)
    sens_b = rng.uniform(0.1, 1.0, size=N_wl)
    norm_a = np.array([
        np.trapz(sens_a[bl_a[i]:br_a[i]], wl[bl_a[i]:br_a[i]])
        for i in range(N_bins_a)
    ])
    norm_b = np.array([
        np.trapz(sens_b[bl_b[i]:br_b[i]], wl[bl_b[i]:br_b[i]])
        for i in range(N_bins_b)
    ])
    sigma_a = np.full(N_bins_a, 1.2)
    sigma_b = np.full(N_bins_b, 1.5)
    data_properties = {
        "datasets": ["A", "B"],
        "instruments": ["JWST_NIRSpec_PRISM", "JWST_NIRISS_SOSS_Ord1"],
        "psf_sigma": np.concatenate([sigma_a, sigma_b]),
        "sens": np.concatenate([sens_a, sens_b]),
        "bin_left": np.concatenate([bl_a, bl_b]),
        "bin_cent": np.concatenate([bc_a, bc_b]),
        "bin_right": np.concatenate([br_a, br_b]),
        "norm": np.concatenate([norm_a, norm_b]),
        "len_data_idx": np.array([0, N_bins_a, N_bins_a + N_bins_b]),
    }
    ours = _instruments.bin_spectrum_to_data(spectrum, wl, data_properties)
    theirs = p_bs(spectrum, wl, data_properties)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_bin_spectrum_to_data_rejects_photometric():
    wl = np.linspace(1.0, 5.0, 100)
    spectrum = np.zeros(100)
    data_properties = {
        "datasets": ["IRAC1"], "instruments": ["IRAC1"],
        "psf_sigma": np.zeros(1), "sens": np.zeros(100),
        "bin_left": np.array([0]), "bin_cent": np.array([50]),
        "bin_right": np.array([99]), "norm": np.array([1.0]),
        "len_data_idx": np.array([0, 1]),
    }
    with pytest.raises(NotImplementedError, match="IRAC1"):
        _instruments.bin_spectrum_to_data(spectrum, wl, data_properties)


# ---------------------------------------------------------------------------
# All offset modes, simple-range and lumped
# ---------------------------------------------------------------------------
def test_apply_offsets_two_datasets_simple():
    ydata = np.arange(10, dtype=float) * 1e-3
    out = _data.apply_offsets(
        ydata, np.array([100.0, -50.0]),
        offsets_applied="two_datasets",
        offset_start=[0, 5], offset_end=[5, 10],
    )
    expected = ydata.copy()
    expected[0:5] -= 100.0 * 1e-6
    expected[5:10] -= -50.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


def test_apply_offsets_three_datasets_simple():
    ydata = np.arange(12, dtype=float) * 1e-3
    out = _data.apply_offsets(
        ydata, np.array([100.0, -50.0, 30.0]),
        offsets_applied="three_datasets",
        offset_start=[0, 4, 8], offset_end=[4, 8, 12],
    )
    expected = ydata.copy()
    expected[0:4] -= 100.0 * 1e-6
    expected[4:8] -= -50.0 * 1e-6
    expected[8:12] -= 30.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


def test_apply_offsets_single_dataset_lumped():
    """Lumped variant: offset_1_start is a non-empty array of segment starts."""
    ydata = np.arange(20, dtype=float) * 1e-3
    out = _data.apply_offsets(
        ydata, np.array([100.0]),
        offsets_applied="single_dataset",
        offset_start=0, offset_end=0,
        offset_1_start=[0, 10], offset_1_end=[5, 15],
    )
    expected = ydata.copy()
    expected[0:5] -= 100.0 * 1e-6
    expected[10:15] -= 100.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


# ---------------------------------------------------------------------------
# loglikelihood full-Gaussian normalisation
# ---------------------------------------------------------------------------
def test_loglikelihood_includes_default_gaussian_norm():
    rng = np.random.default_rng(11)
    n = 20
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = np.full(n, 1e-4)
    ll = _data.loglikelihood(ymodel, ydata, err_data)
    expected = (
        (-0.5 * ((ymodel - ydata) / err_data) ** 2).sum()
        + (-0.5 * np.log(2.0 * np.pi * err_data ** 2)).sum()
    )
    np.testing.assert_allclose(ll, expected, atol=0)


def test_loglikelihood_with_offset_and_inflation():
    """Combined offset + Line15 error inflation matches manual POSEIDON-style calc."""
    rng = np.random.default_rng(13)
    n = 12
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = np.full(n, 1e-4)
    offset_params = np.array([50.0])
    err_inflation_params = np.array([-8.0])
    ll = _data.loglikelihood(
        ymodel, ydata, err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied="single_dataset",
        error_inflation="Line15",
        offset_start=0, offset_end=n,
    )
    ydata_adj = ydata - offset_params[0] * 1e-6
    err_eff_sq = err_data ** 2 + 10.0 ** err_inflation_params[0]
    norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    chi2 = (-0.5 * (ymodel - ydata_adj) ** 2 / err_eff_sq).sum()
    np.testing.assert_allclose(ll, chi2 + norm_log, atol=0)
