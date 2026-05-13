"""Phase 8 tests for _instruments + _data against POSEIDON."""

import numpy as np
import pytest

from jaxposeidon import _data, _instruments

_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


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
    np.testing.assert_allclose(ll, expected, atol=0, rtol=0)


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
        _trapz(sens_a[bl_a[i]:br_a[i]], wl[bl_a[i]:br_a[i]])
        for i in range(N_bins_a)
    ])
    norm_b = np.array([
        _trapz(sens_b[bl_b[i]:br_b[i]], wl[bl_b[i]:br_b[i]])
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
    np.testing.assert_allclose(ll, expected, atol=0, rtol=0)


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
    np.testing.assert_allclose(ll, chi2 + norm_log, atol=0, rtol=0)


def test_apply_offsets_two_datasets_lumped():
    ydata = np.arange(20, dtype=float) * 1e-3
    out = _data.apply_offsets(
        ydata, np.array([100.0, -50.0]),
        offsets_applied="two_datasets",
        offset_start=0, offset_end=0,
        offset_1_start=np.array([0, 10]), offset_1_end=np.array([3, 13]),
        offset_2_start=np.array([5, 15]), offset_2_end=np.array([8, 18]),
    )
    expected = ydata.copy()
    expected[0:3] -= 100.0 * 1e-6
    expected[10:13] -= 100.0 * 1e-6
    expected[5:8] -= -50.0 * 1e-6
    expected[15:18] -= -50.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


def test_apply_offsets_three_datasets_lumped():
    ydata = np.arange(30, dtype=float) * 1e-3
    out = _data.apply_offsets(
        ydata, np.array([100.0, -50.0, 30.0]),
        offsets_applied="three_datasets",
        offset_start=0, offset_end=0,
        offset_1_start=np.array([0, 10]), offset_1_end=np.array([3, 13]),
        offset_2_start=np.array([5, 15]), offset_2_end=np.array([8, 18]),
        offset_3_start=np.array([20, 25]), offset_3_end=np.array([23, 28]),
    )
    expected = ydata.copy()
    expected[0:3] -= 100.0 * 1e-6
    expected[10:13] -= 100.0 * 1e-6
    expected[5:8] -= -50.0 * 1e-6
    expected[15:18] -= -50.0 * 1e-6
    expected[20:23] -= 30.0 * 1e-6
    expected[25:28] -= 30.0 * 1e-6
    np.testing.assert_array_equal(out, expected)


def test_compute_instrument_indices_replicates_poseidon_formula():
    """Replicates the reference-data-independent slice of POSEIDON's
    ``init_instrument`` (``POSEIDON/instrument.py:191-318``) on a precomputed
    sensitivity grid.

    NOTE: This test does NOT invoke POSEIDON. POSEIDON's ``init_instrument``
    dispatches on instrument-specific reference_data sensitivity files (deferred
    to Phase 9); the spectroscopic per-bin index/σ/norm computation we use here
    is the portion of that routine that has no reference_data dependency. The
    "POSEIDON reference computation" block below is a line-for-line transcript
    of ``init_instrument``'s per-bin loop.
    """
    rng = np.random.default_rng(3)
    N_wl, N_bins = 600, 15
    wl = np.linspace(1.0, 5.0, N_wl)
    wl_data = np.linspace(1.5, 4.5, N_bins)
    half_width = np.full(N_bins, 0.04)
    sensitivity = rng.uniform(0.1, 1.0, size=N_wl)
    fwhm_um = np.full(N_bins, 0.02)
    sigma, bl, bc, br, norm = _instruments.compute_instrument_indices(
        wl, wl_data, half_width, sensitivity, fwhm_um,
    )
    # POSEIDON reference computation
    p_bl = np.zeros(N_bins, dtype=np.int64)
    p_bc = np.zeros(N_bins, dtype=np.int64)
    p_br = np.zeros(N_bins, dtype=np.int64)
    p_sig = np.zeros(N_bins)
    p_norm = np.zeros(N_bins)
    sigma_um = 0.424661 * fwhm_um
    for n in range(N_bins):
        p_bl[n] = int(np.argmin(np.abs(wl - (wl_data[n] - half_width[n]))))
        p_bc[n] = int(np.argmin(np.abs(wl - wl_data[n])))
        p_br[n] = int(np.argmin(np.abs(wl - (wl_data[n] + half_width[n]))))
        dwl = 0.5 * (wl[p_bc[n] + 1] - wl[p_bc[n] - 1])
        p_sig[n] = sigma_um[n] / dwl
        p_norm[n] = _trapz(
            sensitivity[p_bl[n]:p_br[n]], wl[p_bl[n]:p_br[n]],
        )
    np.testing.assert_array_equal(bl, p_bl)
    np.testing.assert_array_equal(bc, p_bc)
    np.testing.assert_array_equal(br, p_br)
    np.testing.assert_array_equal(sigma, p_sig)
    np.testing.assert_array_equal(norm, p_norm)


def test_loglikelihood_replicates_poseidon_formula_combined():
    """End-to-end likelihood matches a line-for-line replication of POSEIDON's
    ``LogLikelihood`` body (``POSEIDON/retrieval.py:1087-1183``).

    NOTE: POSEIDON's ``LogLikelihood`` is defined as a closure inside
    ``run_retrieval`` (capturing data_properties, model, priors, sampler state)
    and cannot be invoked in isolation without the full retrieval scaffold.
    This test therefore replicates the arithmetic of lines 1087-1183 below
    rather than calling POSEIDON. End-to-end POSEIDON parity at the
    ``run_retrieval`` level is a Phase 9/10 work item.
    """
    rng = np.random.default_rng(17)
    n = 24
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = rng.uniform(8e-5, 2e-4, size=n)
    offset_params = np.array([75.0, -40.0])
    err_inflation_params = np.array([-8.5, 0.2])
    ln_prior_TP = -1.234
    ll = _data.loglikelihood(
        ymodel, ydata, err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied="two_datasets",
        error_inflation="Line15+Piette20",
        offset_start=[0, n // 2], offset_end=[n // 2, n],
        ln_prior_TP=ln_prior_TP,
    )
    # POSEIDON-style hand computation
    ydata_adj = ydata.copy()
    ydata_adj[0:n // 2] -= offset_params[0] * 1e-6
    ydata_adj[n // 2:n] -= offset_params[1] * 1e-6
    err_eff_sq = (
        err_data ** 2
        + 10.0 ** err_inflation_params[0]
        + (err_inflation_params[1] * ymodel) ** 2
    )
    norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    chi2 = (-0.5 * (ymodel - ydata_adj) ** 2 / err_eff_sq).sum()
    np.testing.assert_allclose(ll, chi2 + norm_log + ln_prior_TP, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# Combinatorial v0-surface loglikelihood coverage
# ---------------------------------------------------------------------------
def _hand_loglikelihood(ymodel, ydata, err_data, offsets_applied, error_inflation,
                        offset_params, err_inflation_params,
                        offset_start, offset_end,
                        offset_1_start=0, offset_1_end=0,
                        offset_2_start=0, offset_2_end=0,
                        offset_3_start=0, offset_3_end=0,
                        ln_prior_TP=0.0):
    """Line-for-line replication of POSEIDON retrieval.py:1087-1183."""
    ydata_adj = ydata.copy()
    if offsets_applied == "single_dataset":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start:offset_end] -= offset_params[0] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n]:offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
    elif offsets_applied == "two_datasets":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start[0]:offset_end[0]] -= offset_params[0] * 1e-6
            ydata_adj[offset_start[1]:offset_end[1]] -= offset_params[1] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n]:offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj[offset_2_start[m]:offset_2_end[m]] -= (
                    offset_params[1] * 1e-6
                )
    elif offsets_applied == "three_datasets":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start[0]:offset_end[0]] -= offset_params[0] * 1e-6
            ydata_adj[offset_start[1]:offset_end[1]] -= offset_params[1] * 1e-6
            ydata_adj[offset_start[2]:offset_end[2]] -= offset_params[2] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n]:offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj[offset_2_start[m]:offset_2_end[m]] -= (
                    offset_params[1] * 1e-6
                )
            for s in range(len(offset_3_start)):
                ydata_adj[offset_3_start[s]:offset_3_end[s]] -= (
                    offset_params[2] * 1e-6
                )

    if error_inflation is None:
        err_eff_sq = err_data ** 2
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Line15":
        err_eff_sq = err_data ** 2 + 10.0 ** err_inflation_params[0]
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Piette20":
        err_eff_sq = err_data ** 2 + (err_inflation_params[0] * ymodel) ** 2
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Line15+Piette20":
        err_eff_sq = (err_data ** 2 + 10.0 ** err_inflation_params[0]
                      + (err_inflation_params[1] * ymodel) ** 2)
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    chi2 = (-0.5 * (ymodel - ydata_adj) ** 2 / err_eff_sq).sum()
    return chi2 + norm_log + ln_prior_TP


_OFFSET_MODES = [None, "single_dataset", "two_datasets", "three_datasets"]
_INFLATION_MODES = [None, "Line15", "Piette20", "Line15+Piette20"]


def _offset_setup(mode, n):
    if mode is None:
        return dict(), np.array([])
    if mode == "single_dataset":
        return dict(offset_start=0, offset_end=n), np.array([100.0])
    if mode == "two_datasets":
        h = n // 2
        return (dict(offset_start=[0, h], offset_end=[h, n]),
                np.array([100.0, -50.0]))
    if mode == "three_datasets":
        t = n // 3
        return (dict(offset_start=[0, t, 2*t], offset_end=[t, 2*t, n]),
                np.array([100.0, -50.0, 30.0]))


def _inflation_setup(mode):
    if mode is None:
        return np.array([])
    if mode == "Line15":
        return np.array([-8.5])
    if mode == "Piette20":
        return np.array([0.2])
    if mode == "Line15+Piette20":
        return np.array([-9.0, 0.15])


@pytest.mark.parametrize("offsets_applied", _OFFSET_MODES)
@pytest.mark.parametrize("error_inflation", _INFLATION_MODES)
def test_loglikelihood_combinatorial_v0_surface(offsets_applied, error_inflation):
    """Cover every (offsets_applied × error_inflation) combination in v0."""
    rng = np.random.default_rng(hash((offsets_applied, error_inflation)) & 0xFFFF)
    n = 24
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = rng.uniform(8e-5, 2e-4, size=n)
    off_kwargs, offset_params = _offset_setup(offsets_applied, n)
    err_inflation_params = _inflation_setup(error_inflation)
    ln_prior_TP = -0.7

    ll = _data.loglikelihood(
        ymodel, ydata, err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        ln_prior_TP=ln_prior_TP,
        **off_kwargs,
    )
    expected = _hand_loglikelihood(
        ymodel, ydata, err_data,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offset_start=off_kwargs.get("offset_start", 0),
        offset_end=off_kwargs.get("offset_end", 0),
        ln_prior_TP=ln_prior_TP,
    )
    np.testing.assert_allclose(ll, expected, atol=0, rtol=0)


@pytest.mark.parametrize("offsets_applied", ["single_dataset", "two_datasets",
                                              "three_datasets"])
@pytest.mark.parametrize("error_inflation", _INFLATION_MODES)
def test_loglikelihood_combinatorial_v0_surface_lumped(offsets_applied,
                                                       error_inflation):
    """Lumped offset ranges through loglikelihood across all v0 configs."""
    rng = np.random.default_rng(hash((offsets_applied, error_inflation, "L"))
                                 & 0xFFFF)
    n = 30
    ydata = rng.uniform(2.5e-3, 3.0e-3, size=n)
    ymodel = ydata + rng.normal(0, 5e-5, size=n)
    err_data = rng.uniform(8e-5, 2e-4, size=n)
    err_inflation_params = _inflation_setup(error_inflation)
    ln_prior_TP = 0.42

    lumped_kwargs = dict(offset_start=0, offset_end=0)
    if offsets_applied == "single_dataset":
        lumped_kwargs.update(
            offset_1_start=np.array([0, 15]),
            offset_1_end=np.array([5, 20]),
        )
        offset_params = np.array([100.0])
    elif offsets_applied == "two_datasets":
        lumped_kwargs.update(
            offset_1_start=np.array([0, 15]),
            offset_1_end=np.array([5, 20]),
            offset_2_start=np.array([5, 20]),
            offset_2_end=np.array([10, 25]),
        )
        offset_params = np.array([100.0, -50.0])
    else:  # three_datasets
        lumped_kwargs.update(
            offset_1_start=np.array([0]),
            offset_1_end=np.array([5]),
            offset_2_start=np.array([10]),
            offset_2_end=np.array([15]),
            offset_3_start=np.array([20]),
            offset_3_end=np.array([25]),
        )
        offset_params = np.array([100.0, -50.0, 30.0])

    ll = _data.loglikelihood(
        ymodel, ydata, err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        ln_prior_TP=ln_prior_TP,
        **lumped_kwargs,
    )
    expected = _hand_loglikelihood(
        ymodel, ydata, err_data,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offset_start=0, offset_end=0,
        ln_prior_TP=ln_prior_TP,
        **{k: v for k, v in lumped_kwargs.items()
            if k not in ("offset_start", "offset_end")},
    )
    np.testing.assert_allclose(ll, expected, atol=0, rtol=0)


def test_loglikelihood_NaN_sentinel_overrides_all_extras():
    """NaN spectrum returns exactly -1e100 regardless of norm/inflation/offsets/prior."""
    n = 12
    ydata = np.full(n, 3e-3)
    err_data = np.full(n, 1e-4)
    ymodel = ydata.copy()
    ymodel[5] = np.nan
    ll = _data.loglikelihood(
        ymodel, ydata, err_data,
        offset_params=np.array([50.0]),
        err_inflation_params=np.array([-8.0, 0.3]),
        offsets_applied="single_dataset",
        error_inflation="Line15+Piette20",
        offset_start=0, offset_end=n,
        norm_log_default=12.34,
        ln_prior_TP=-5.67,
    )
    assert ll == -1.0e100
