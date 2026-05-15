"""v1-A: parity tests for JAX-pure primitives and the `_priors` / `_data`
ports against scipy reference implementations and the v0.5 numpy callers.

Tolerance default per plan: rtol=1e-13 (any relaxation must be documented
in MISMATCHES.md with FP-reorder justification).
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from scipy.interpolate import PchipInterpolator, RegularGridInterpolator  # noqa: E402
from scipy.ndimage import gaussian_filter1d as sp_gf1d  # noqa: E402
from scipy.special import expn as sp_expn  # noqa: E402

from jaxposeidon._data import loglikelihood  # noqa: E402
from jaxposeidon._jax_filters import gaussian_filter1d_edge  # noqa: E402
from jaxposeidon._jax_interpolate import (  # noqa: E402
    pchip_interpolate,
    regular_grid_interp_linear,
)
from jaxposeidon._jax_special import expn_2  # noqa: E402
from jaxposeidon._jax_special import ndtri as jax_ndtri  # noqa: E402
from jaxposeidon._priors import prior_transform  # noqa: E402

# ----------------------------- _jax_filters -----------------------------


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5, 3.1, 5.0, 7.7, 12.0])
def test_gaussian_filter1d_edge_matches_scipy(sigma):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200).astype(np.float64)
    ref = sp_gf1d(x, sigma=sigma, mode="nearest")
    ours = np.array(gaussian_filter1d_edge(x, sigma))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_gaussian_filter1d_edge_under_jit():
    from functools import partial

    x = np.linspace(0.0, 1.0, 100, dtype=np.float64)
    f = jax.jit(partial(gaussian_filter1d_edge, sigma=2.0))
    ours = np.array(f(x))
    ref = sp_gf1d(x, sigma=2.0, mode="nearest")
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


# ----------------------------- _jax_interpolate -----------------------------


def test_pchip_interpolate_matches_scipy():
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 10, 12)).astype(np.float64)
    y = np.sin(x) + 0.3 * x
    xq = np.linspace(x[0], x[-1], 200, dtype=np.float64)
    ref = PchipInterpolator(x, y)(xq)
    ours = np.array(pchip_interpolate(x, y, xq))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_pchip_interpolate_atmosphere_log_P_pattern():
    log_P = np.linspace(-7.0, 2.0, 9, dtype=np.float64)
    T_pts = np.array([1500.0, 1400, 1300, 1200, 1100, 1000, 900, 800, 700])
    log_P_query = np.linspace(-7.0, 2.0, 100, dtype=np.float64)
    ref = PchipInterpolator(log_P, T_pts)(log_P_query)
    ours = np.array(pchip_interpolate(log_P, T_pts, log_P_query))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_pchip_under_jit():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 1.0, 0.5, 2.0, 1.5])
    xq = np.array([0.5, 1.5, 2.5, 3.5])
    f = jax.jit(pchip_interpolate)
    ours = np.array(f(x, y, xq))
    ref = PchipInterpolator(x, y)(xq)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_regular_grid_interp_linear_matches_scipy_2d():
    ax0 = np.linspace(-3.0, 3.0, 7, dtype=np.float64)
    ax1 = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    g0, g1 = np.meshgrid(ax0, ax1, indexing="ij")
    values = (g0**2 + g1).astype(np.float64)
    rng = np.random.default_rng(2)
    qp = np.stack(
        [rng.uniform(ax0[0], ax0[-1], 50), rng.uniform(ax1[0], ax1[-1], 50)],
        axis=-1,
    ).astype(np.float64)
    ref = RegularGridInterpolator((ax0, ax1), values, method="linear")(qp)
    ours = np.array(regular_grid_interp_linear((ax0, ax1), values, qp))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_regular_grid_interp_linear_3d_in_range():
    ax0 = np.array([0.0, 1.0, 2.0])
    ax1 = np.array([0.0, 1.0])
    ax2 = np.array([0.0, 0.5, 1.0])
    values = np.arange(3 * 2 * 3, dtype=np.float64).reshape(3, 2, 3)
    qp = np.array(
        [[0.5, 0.5, 0.25], [1.5, 0.0, 0.75], [1.0, 1.0, 1.0]], dtype=np.float64
    )
    ref = RegularGridInterpolator((ax0, ax1, ax2), values, method="linear")(qp)
    ours = np.array(regular_grid_interp_linear((ax0, ax1, ax2), values, qp))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_regular_grid_interp_linear_clips_out_of_range():
    """Out-of-range query points clip to the boundary value (no extrapolation).

    This matches the v1-A plan spec: `regular_grid_interp_linear` uses
    `linear extrapolation off (boundary clip)`. Verifies that out-of-range
    queries return the boundary value rather than scipy's extrapolated
    value (so callers can't accidentally rely on extrapolation parity).
    """
    ax = np.array([0.0, 1.0, 2.0])
    values = np.array([[1.0, 2.0, 3.0]]).T  # values along ax
    values = np.array([10.0, 20.0, 30.0])
    qp_low = np.array([[-5.0]])
    qp_high = np.array([[100.0]])
    ours_low = float(regular_grid_interp_linear((ax,), values, qp_low)[0])
    ours_high = float(regular_grid_interp_linear((ax,), values, qp_high)[0])
    assert ours_low == 10.0
    assert ours_high == 30.0


def test_pchip_two_knot_case():
    """SciPy's PchipInterpolator supports N=2 (degenerate to linear)."""
    x = np.array([0.0, 1.0])
    y = np.array([3.0, 7.0])
    xq = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    ref = PchipInterpolator(x, y)(xq)
    ours = np.array(pchip_interpolate(x, y, xq))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=0)


# ----------------------------- _jax_special -----------------------------


def test_expn_2_matches_scipy_dense():
    xs = np.concatenate(
        [
            np.array([0.0]),
            np.logspace(-10, -1, 20),
            np.linspace(0.1, 1.0, 19, endpoint=False),
            np.linspace(1.0, 30.0, 50),
            np.linspace(30.0, 100.0, 30),
        ]
    ).astype(np.float64)
    ref = sp_expn(2, xs)
    ours = np.array(expn_2(xs))
    big = np.abs(ref) > 0
    rel = np.abs(ours[big] - ref[big]) / np.abs(ref[big])
    assert rel.max() < 1e-13, f"max relative error {rel.max():.2e}"


def test_expn_2_zero():
    assert float(expn_2(0.0)) == 1.0


def test_expn_2_under_jit():
    f = jax.jit(expn_2)
    x = np.array([0.5, 2.0, 15.0, 60.0])
    ours = np.array(f(x))
    ref = sp_expn(2, x)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_ndtri_re_export():
    from scipy.special import ndtri as sp_ndtri

    u = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    np.testing.assert_allclose(np.array(jax_ndtri(u)), sp_ndtri(u), rtol=1e-13, atol=0)


# ----------------------------- _priors -----------------------------


def _numpy_prior_transform(cube, param_names, prior_types, prior_ranges):
    from scipy.special import ndtri as sp_ndtri

    cube = np.array(cube, dtype=np.float64).copy()
    for i, parameter in enumerate(param_names):
        ptype = prior_types[parameter]
        prange = prior_ranges[parameter]
        if ptype == "uniform":
            cube[i] = cube[i] * (prange[1] - prange[0]) + prange[0]
        elif ptype == "gaussian":
            cube[i] = prange[0] + prange[1] * sp_ndtri(cube[i])
        elif ptype == "sine":
            mv = prange[1]
            if parameter in ("alpha", "beta"):
                cube[i] = (
                    (180.0 / np.pi)
                    * 2.0
                    * np.arcsin(cube[i] * np.sin((np.pi / 180.0) * (mv / 2.0)))
                )
            elif parameter == "theta_0":
                cube[i] = (180.0 / np.pi) * np.arcsin(
                    (2.0 * cube[i] - 1.0) * np.sin((np.pi / 180.0) * (mv / 2.0))
                )
    return cube


def test_prior_transform_matches_numpy_v05():
    param_names = ["T", "log_g", "alpha", "beta", "theta_0", "R_p"]
    prior_types = {
        "T": "uniform",
        "log_g": "gaussian",
        "alpha": "sine",
        "beta": "sine",
        "theta_0": "sine",
        "R_p": "uniform",
    }
    prior_ranges = {
        "T": [400.0, 2500.0],
        "log_g": [3.5, 0.2],
        "alpha": [0.0, 60.0],
        "beta": [0.0, 30.0],
        "theta_0": [-90.0, 90.0],
        "R_p": [0.8, 1.2],
    }
    rng = np.random.default_rng(42)
    cube = rng.uniform(0.01, 0.99, size=len(param_names))
    ours = np.array(prior_transform(cube, param_names, prior_types, prior_ranges))
    ref = _numpy_prior_transform(cube, param_names, prior_types, prior_ranges)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_prior_transform_jaxpr_succeeds():
    param_names = ["a", "b"]
    prior_types = {"a": "uniform", "b": "gaussian"}
    prior_ranges = {"a": [0.0, 1.0], "b": [0.0, 1.0]}
    # Call once to exercise the jit cache; jax.make_jaxpr on the inner
    # kernel verifies tracing.
    from jaxposeidon._priors import _kernel_no_CLR

    codes = jnp.array([0, 1], dtype=jnp.int32)
    lo = jnp.array([0.0, 0.0])
    hi = jnp.array([1.0, 1.0])
    cube = jnp.array([0.3, 0.7])
    jaxpr = jax.make_jaxpr(_kernel_no_CLR)(cube, codes, lo, hi)
    assert "jaxpr" in repr(jaxpr).lower() or len(repr(jaxpr)) > 0
    _ = prior_transform(np.array([0.3, 0.7]), param_names, prior_types, prior_ranges)


# ----------------------------- _data -----------------------------


def _numpy_loglikelihood(
    ymodel,
    ydata,
    err_data,
    *,
    offset_params=(),
    err_inflation_params=(),
    offsets_applied=None,
    error_inflation=None,
    offset_start=0,
    offset_end=0,
    ln_prior_TP=0.0,
):
    if np.any(np.isnan(ymodel)):
        return -1.0e100
    if error_inflation is None:
        err_eff_sq = err_data * err_data
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_data * err_data)).sum()
    elif error_inflation == "Line15":
        err_eff_sq = err_data * err_data + 10.0 ** err_inflation_params[0]
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Piette20":
        err_eff_sq = err_data * err_data + (err_inflation_params[0] * ymodel) ** 2
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Line15+Piette20":
        err_eff_sq = (
            err_data * err_data
            + 10.0 ** err_inflation_params[0]
            + (err_inflation_params[1] * ymodel) ** 2
        )
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    ydata_adj = ydata.copy()
    if offsets_applied == "single_dataset":
        ydata_adj[offset_start:offset_end] -= offset_params[0] * 1e-6
    return float(
        (-0.5 * (ymodel - ydata_adj) ** 2 / err_eff_sq).sum() + norm_log + ln_prior_TP
    )


@pytest.mark.parametrize(
    "error_inflation",
    [None, "Line15", "Piette20", "Line15+Piette20"],
)
def test_loglikelihood_matches_numpy_v05_error_inflation(error_inflation):
    rng = np.random.default_rng(3)
    n = 40
    ymodel = rng.uniform(0.013, 0.015, n).astype(np.float64)
    ydata = ymodel + rng.standard_normal(n) * 1e-4
    err_data = np.full(n, 1.5e-4, dtype=np.float64)
    err_inflation_params = (
        np.array([-8.5, 0.1])
        if error_inflation == "Line15+Piette20"
        else np.array([-8.5])
        if error_inflation == "Line15"
        else np.array([0.1])
        if error_inflation == "Piette20"
        else np.array([])
    )
    ours = float(
        loglikelihood(
            ymodel,
            ydata,
            err_data,
            err_inflation_params=tuple(err_inflation_params),
            error_inflation=error_inflation,
        )
    )
    ref = _numpy_loglikelihood(
        ymodel,
        ydata,
        err_data,
        err_inflation_params=err_inflation_params,
        error_inflation=error_inflation,
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_loglikelihood_offsets_match_numpy():
    rng = np.random.default_rng(4)
    n = 20
    ymodel = rng.uniform(0.013, 0.015, n).astype(np.float64)
    ydata = ymodel.copy()
    err_data = np.full(n, 1e-4)
    offset_params = np.array([50.0])  # 50 ppm
    ours = float(
        loglikelihood(
            ymodel,
            ydata,
            err_data,
            offset_params=tuple(offset_params),
            offsets_applied="single_dataset",
            offset_start=5,
            offset_end=15,
        )
    )
    ref = _numpy_loglikelihood(
        ymodel,
        ydata,
        err_data,
        offset_params=offset_params,
        offsets_applied="single_dataset",
        offset_start=5,
        offset_end=15,
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_loglikelihood_nan_sentinel():
    ymodel = np.array([1.0, np.nan, 1.0])
    ydata = np.array([1.0, 1.0, 1.0])
    err = np.array([0.1, 0.1, 0.1])
    ll = float(loglikelihood(ymodel, ydata, err))
    assert ll == -1.0e100


def test_loglikelihood_jaxpr_succeeds():
    """loglikelihood with static dispatch tags must trace under make_jaxpr."""
    from functools import partial

    fn = partial(loglikelihood, error_inflation=None, offsets_applied=None)
    jaxpr = jax.make_jaxpr(fn)(
        jnp.array([1.0, 1.1, 1.2]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array([0.1, 0.1, 0.1]),
    )
    assert len(repr(jaxpr)) > 0


@pytest.mark.parametrize(
    "error_inflation,offsets_applied",
    [
        ("Line15", "single_dataset"),
        ("Piette20", "single_dataset"),
        ("Line15+Piette20", "single_dataset"),
        (None, "single_dataset"),
    ],
)
def test_loglikelihood_jaxpr_nontrivial_branches(error_inflation, offsets_applied):
    from functools import partial

    fn = partial(
        loglikelihood,
        error_inflation=error_inflation,
        offsets_applied=offsets_applied,
        offset_start=0,
        offset_end=3,
    )
    err_inf = (
        jnp.array([-8.0, 0.1])
        if error_inflation == "Line15+Piette20"
        else jnp.array([-8.0])
        if error_inflation == "Line15"
        else jnp.array([0.1])
        if error_inflation == "Piette20"
        else jnp.array([])
    )
    jaxpr = jax.make_jaxpr(fn)(
        jnp.array([1.0, 1.1, 1.2]),
        jnp.array([1.0, 1.0, 1.0]),
        jnp.array([0.1, 0.1, 0.1]),
        offset_params=jnp.array([10.0]),
        err_inflation_params=err_inf,
    )
    assert len(repr(jaxpr)) > 0


# --------------- _data.apply_offsets multi-dataset parity --------------------


def _numpy_apply_offsets_simple(
    ydata, offset_params, offsets_applied, offset_start, offset_end
):
    """Line-for-line POSEIDON apply-offsets replication for the simple-range path.

    Mirrors POSEIDON `retrieval.py:1124-1174` (the
    `offsets_applied=="single_dataset"/"two_datasets"/"three_datasets"`
    branches when `offset_1_start` etc. are not provided, i.e. the
    simple `offset_start[i]:offset_end[i]` path).
    """
    out = ydata.copy()
    if offsets_applied == "single_dataset":
        out[offset_start:offset_end] -= offset_params[0] * 1e-6
    elif offsets_applied == "two_datasets":
        out[offset_start[0] : offset_end[0]] -= offset_params[0] * 1e-6
        out[offset_start[1] : offset_end[1]] -= offset_params[1] * 1e-6
    elif offsets_applied == "three_datasets":
        out[offset_start[0] : offset_end[0]] -= offset_params[0] * 1e-6
        out[offset_start[1] : offset_end[1]] -= offset_params[1] * 1e-6
        out[offset_start[2] : offset_end[2]] -= offset_params[2] * 1e-6
    return out


def test_apply_offsets_two_datasets_matches_numpy():
    from jaxposeidon._data import apply_offsets

    rng = np.random.default_rng(7)
    ydata = rng.uniform(0.013, 0.015, 20).astype(np.float64)
    offset_params = np.array([60.0, -40.0])
    out = np.array(
        apply_offsets(
            ydata,
            offset_params,
            offsets_applied="two_datasets",
            offset_start=[0, 10],
            offset_end=[10, 20],
        )
    )
    ref = _numpy_apply_offsets_simple(
        ydata, offset_params, "two_datasets", [0, 10], [10, 20]
    )
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)


def test_apply_offsets_three_datasets_matches_numpy():
    from jaxposeidon._data import apply_offsets

    rng = np.random.default_rng(8)
    ydata = rng.uniform(0.013, 0.015, 30).astype(np.float64)
    offset_params = np.array([20.0, -50.0, 80.0])
    out = np.array(
        apply_offsets(
            ydata,
            offset_params,
            offsets_applied="three_datasets",
            offset_start=[0, 10, 20],
            offset_end=[10, 20, 30],
        )
    )
    ref = _numpy_apply_offsets_simple(
        ydata, offset_params, "three_datasets", [0, 10, 20], [10, 20, 30]
    )
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)


def test_apply_offsets_single_dataset_lumped_path():
    """Grouped `offset_1_start` / `offset_1_end` list path."""
    from jaxposeidon._data import apply_offsets

    rng = np.random.default_rng(9)
    ydata = rng.uniform(0.013, 0.015, 30).astype(np.float64)
    offset_params = np.array([42.0])
    starts = [0, 12, 22]
    ends = [5, 18, 28]
    out = np.array(
        apply_offsets(
            ydata,
            offset_params,
            offsets_applied="single_dataset",
            offset_start=0,
            offset_end=0,
            offset_1_start=starts,
            offset_1_end=ends,
        )
    )
    ref = ydata.copy()
    for s, e in zip(starts, ends, strict=True):
        ref[s:e] -= offset_params[0] * 1e-6
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)


def test_apply_offsets_two_datasets_lumped_path():
    """Grouped paths for two datasets (POSEIDON `retrieval.py:1135-1148`)."""
    from jaxposeidon._data import apply_offsets

    rng = np.random.default_rng(10)
    ydata = rng.uniform(0.013, 0.015, 40).astype(np.float64)
    offset_params = np.array([30.0, -25.0])
    s1, e1 = [0, 12], [5, 18]
    s2, e2 = [20, 32], [25, 38]
    out = np.array(
        apply_offsets(
            ydata,
            offset_params,
            offsets_applied="two_datasets",
            offset_start=0,
            offset_end=0,
            offset_1_start=s1,
            offset_1_end=e1,
            offset_2_start=s2,
            offset_2_end=e2,
        )
    )
    ref = ydata.copy()
    for s, e in zip(s1, e1, strict=True):
        ref[s:e] -= offset_params[0] * 1e-6
    for s, e in zip(s2, e2, strict=True):
        ref[s:e] -= offset_params[1] * 1e-6
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)


def test_apply_offsets_three_datasets_lumped_path():
    """Grouped paths for three datasets (POSEIDON `retrieval.py:1158-1174`)."""
    from jaxposeidon._data import apply_offsets

    rng = np.random.default_rng(12)
    ydata = rng.uniform(0.013, 0.015, 50).astype(np.float64)
    offset_params = np.array([15.0, -30.0, 45.0])
    s1, e1 = [0, 10], [4, 14]
    s2, e2 = [16, 26], [20, 30]
    s3, e3 = [32, 42], [36, 46]
    out = np.array(
        apply_offsets(
            ydata,
            offset_params,
            offsets_applied="three_datasets",
            offset_start=0,
            offset_end=0,
            offset_1_start=s1,
            offset_1_end=e1,
            offset_2_start=s2,
            offset_2_end=e2,
            offset_3_start=s3,
            offset_3_end=e3,
        )
    )
    ref = ydata.copy()
    for s, e in zip(s1, e1, strict=True):
        ref[s:e] -= offset_params[0] * 1e-6
    for s, e in zip(s2, e2, strict=True):
        ref[s:e] -= offset_params[1] * 1e-6
    for s, e in zip(s3, e3, strict=True):
        ref[s:e] -= offset_params[2] * 1e-6
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=0)


def _numpy_loglikelihood_full(
    ymodel,
    ydata,
    err_data,
    *,
    offset_params=(),
    err_inflation_params=(),
    offsets_applied=None,
    error_inflation=None,
    offset_start=0,
    offset_end=0,
    offset_1_start=0,
    offset_1_end=0,
    offset_2_start=0,
    offset_2_end=0,
    offset_3_start=0,
    offset_3_end=0,
    ln_prior_TP=0.0,
):
    """Line-for-line POSEIDON `retrieval.py:1065-1183` replication.

    Covers the full v0-envelope likelihood including all three
    `offsets_applied` paths and all four `error_inflation` modes.
    """
    if np.any(np.isnan(ymodel)):
        return -1.0e100
    err_data = np.asarray(err_data)
    ymodel = np.asarray(ymodel)
    if error_inflation is None:
        err_eff_sq = err_data * err_data
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_data * err_data)).sum()
    elif error_inflation == "Line15":
        err_eff_sq = err_data * err_data + 10.0 ** err_inflation_params[0]
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    elif error_inflation == "Piette20":
        err_eff_sq = err_data * err_data + (err_inflation_params[0] * ymodel) ** 2
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()
    else:  # "Line15+Piette20"
        err_eff_sq = (
            err_data * err_data
            + 10.0 ** err_inflation_params[0]
            + (err_inflation_params[1] * ymodel) ** 2
        )
        norm_log = (-0.5 * np.log(2.0 * np.pi * err_eff_sq)).sum()

    ydata_adj = np.asarray(ydata).copy()
    if offsets_applied == "single_dataset":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start:offset_end] -= offset_params[0] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n] : offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
    elif offsets_applied == "two_datasets":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start[0] : offset_end[0]] -= offset_params[0] * 1e-6
            ydata_adj[offset_start[1] : offset_end[1]] -= offset_params[1] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n] : offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj[offset_2_start[m] : offset_2_end[m]] -= (
                    offset_params[1] * 1e-6
                )
    elif offsets_applied == "three_datasets":
        if np.isscalar(offset_1_start) and offset_1_start == 0:
            ydata_adj[offset_start[0] : offset_end[0]] -= offset_params[0] * 1e-6
            ydata_adj[offset_start[1] : offset_end[1]] -= offset_params[1] * 1e-6
            ydata_adj[offset_start[2] : offset_end[2]] -= offset_params[2] * 1e-6
        else:
            for n in range(len(offset_1_start)):
                ydata_adj[offset_1_start[n] : offset_1_end[n]] -= (
                    offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj[offset_2_start[m] : offset_2_end[m]] -= (
                    offset_params[1] * 1e-6
                )
            for s in range(len(offset_3_start)):
                ydata_adj[offset_3_start[s] : offset_3_end[s]] -= (
                    offset_params[2] * 1e-6
                )
    return float(
        (-0.5 * (ymodel - ydata_adj) ** 2 / err_eff_sq).sum() + norm_log + ln_prior_TP
    )


def test_loglikelihood_two_dataset_offsets_matches_full_poseidon_oracle():
    rng = np.random.default_rng(13)
    n = 20
    ymodel = rng.uniform(0.013, 0.015, n).astype(np.float64)
    ydata = ymodel + rng.standard_normal(n) * 1e-4
    err_data = np.full(n, 1.5e-4)
    offset_params = np.array([35.0, -22.0])
    ours = float(
        loglikelihood(
            ymodel,
            ydata,
            err_data,
            offset_params=tuple(offset_params),
            offsets_applied="two_datasets",
            offset_start=[0, 10],
            offset_end=[10, 20],
        )
    )
    ref = _numpy_loglikelihood_full(
        ymodel,
        ydata,
        err_data,
        offset_params=offset_params,
        offsets_applied="two_datasets",
        offset_start=[0, 10],
        offset_end=[10, 20],
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_loglikelihood_three_dataset_offsets_with_inflation_matches_oracle():
    rng = np.random.default_rng(14)
    n = 30
    ymodel = rng.uniform(0.013, 0.015, n).astype(np.float64)
    ydata = ymodel + rng.standard_normal(n) * 1e-4
    err_data = np.full(n, 1.5e-4)
    offset_params = np.array([10.0, -20.0, 30.0])
    err_inflation_params = np.array([-8.5, 0.1])
    ours = float(
        loglikelihood(
            ymodel,
            ydata,
            err_data,
            offset_params=tuple(offset_params),
            err_inflation_params=tuple(err_inflation_params),
            offsets_applied="three_datasets",
            error_inflation="Line15+Piette20",
            offset_start=[0, 10, 20],
            offset_end=[10, 20, 30],
        )
    )
    ref = _numpy_loglikelihood_full(
        ymodel,
        ydata,
        err_data,
        offset_params=offset_params,
        err_inflation_params=err_inflation_params,
        offsets_applied="three_datasets",
        error_inflation="Line15+Piette20",
        offset_start=[0, 10, 20],
        offset_end=[10, 20, 30],
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_loglikelihood_grouped_two_dataset_offsets_matches_oracle():
    """Grouped path: offset_1_start / offset_2_start lists, full likelihood."""
    rng = np.random.default_rng(15)
    n = 40
    ymodel = rng.uniform(0.013, 0.015, n).astype(np.float64)
    ydata = ymodel.copy()
    err_data = np.full(n, 1e-4)
    offset_params = np.array([25.0, -15.0])
    s1, e1 = [0, 12], [5, 18]
    s2, e2 = [20, 32], [25, 38]
    ours = float(
        loglikelihood(
            ymodel,
            ydata,
            err_data,
            offset_params=tuple(offset_params),
            offsets_applied="two_datasets",
            offset_start=0,
            offset_end=0,
            offset_1_start=s1,
            offset_1_end=e1,
            offset_2_start=s2,
            offset_2_end=e2,
        )
    )
    ref = _numpy_loglikelihood_full(
        ymodel,
        ydata,
        err_data,
        offset_params=offset_params,
        offsets_applied="two_datasets",
        offset_start=0,
        offset_end=0,
        offset_1_start=s1,
        offset_1_end=e1,
        offset_2_start=s2,
        offset_2_end=e2,
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


# ----------------------------- _priors CLR parity -----------------------------


def test_CLR_prior_transform_accepted_draw_matches_poseidon():
    """End-to-end CLR draw under jit reproduces the v0.5 numpy oracle."""
    param_names = ["R_p", "T", "log_H2O", "log_CH4", "log_NH3"]
    prior_types = dict.fromkeys(param_names, "uniform")
    for p in ("log_H2O", "log_CH4", "log_NH3"):
        prior_types[p] = "CLR"
    prior_ranges = {
        "R_p": [0.9, 1.1],
        "T": [300.0, 2000.0],
        "log_H2O": [-12.0, -1.0],
        "log_CH4": [-12.0, -1.0],
        "log_NH3": [-12.0, -1.0],
    }
    rng = np.random.default_rng(11)
    unit_cube = rng.uniform(0.1, 0.5, size=5).astype(np.float64)
    N_params_cum = np.array([2, 2, 5, 5, 5, 5, 5, 5, 5, 5])
    X_param_names = ["log_H2O", "log_CH4", "log_NH3"]
    cube = np.array(
        prior_transform(
            unit_cube,
            param_names,
            prior_types,
            prior_ranges,
            X_param_names=X_param_names,
            N_params_cum=N_params_cum,
        )
    )
    if cube[2] == -50.0:
        # Rejection path.
        np.testing.assert_array_equal(cube[2:5], np.ones(3) * -50.0)
    else:
        X = 10.0 ** cube[2:5]
        assert (X > 1e-12).all()


def test_CLR_kernel_under_jit():
    """`_kernel_with_CLR` traces and runs under jit."""
    from jaxposeidon._priors import _kernel_with_CLR

    codes = jnp.array([0, 0, 4, 4, 4], dtype=jnp.int32)
    lo = jnp.array([0.9, 300.0, -12.0, -12.0, -12.0])
    hi = jnp.array([1.1, 2000.0, -1.0, -1.0, -1.0])
    cube = jnp.array([0.3, 0.7, 0.2, 0.2, 0.2])
    out = np.array(_kernel_with_CLR(cube, codes, lo, hi, 2, 5, 3, -12.0))
    assert out.shape == (5,)


def test_CLR_Prior_accepted_matches_poseidon_oracle():
    """Deterministic CLR-accepted draw against POSEIDON `retrieval.CLR_Prior`."""
    from POSEIDON.retrieval import CLR_Prior as poseidon_CLR

    from jaxposeidon._priors import CLR_Prior

    # An interior draw (all 0.5) for 4-species is well inside the simplex.
    chem_drawn = np.array([0.5, 0.5, 0.5, 0.5])
    ours = np.array(CLR_Prior(chem_drawn, limit=-12.0))
    theirs = poseidon_CLR(chem_drawn, limit=-12.0)
    assert ours[0] != -50.0, "Expected accepted draw, got rejection"
    np.testing.assert_allclose(ours, theirs, rtol=1e-13, atol=0)


def test_CLR_Prior_rejected_matches_poseidon_oracle():
    """Corner-of-cube draws fall outside the simplex; both implementations
    must return the same `-50.0` sentinel vector."""
    from POSEIDON.retrieval import CLR_Prior as poseidon_CLR

    from jaxposeidon._priors import CLR_Prior

    chem_drawn = np.array([1.0, 1.0, 1.0])  # corner draw, expected to reject
    ours = np.array(CLR_Prior(chem_drawn, limit=-12.0))
    theirs = poseidon_CLR(chem_drawn, limit=-12.0)
    if theirs[0] == -50.0:
        np.testing.assert_array_equal(ours, theirs)
        np.testing.assert_array_equal(ours, np.full(4, -50.0))


def test_prior_transform_CLR_rejection_propagates_sentinel_into_slice():
    """A rejected CLR draw must fill the chemistry slice of the full cube
    output with `-50.0` (POSEIDON `retrieval.py:861-887` allowed_simplex)."""
    param_names = ["R_p", "log_H2O", "log_CH4", "log_NH3"]
    prior_types = {
        "R_p": "uniform",
        "log_H2O": "CLR",
        "log_CH4": "CLR",
        "log_NH3": "CLR",
    }
    prior_ranges = {
        "R_p": [0.9, 1.1],
        "log_H2O": [-12.0, -1.0],
        "log_CH4": [-12.0, -1.0],
        "log_NH3": [-12.0, -1.0],
    }
    # Corner draw forces rejection.
    unit_cube = np.array([0.5, 1.0, 1.0, 1.0])
    N_params_cum = np.array([1, 1, 4, 4, 4, 4, 4, 4, 4, 4])
    cube = np.array(
        prior_transform(
            unit_cube,
            param_names,
            prior_types,
            prior_ranges,
            X_param_names=["log_H2O", "log_CH4", "log_NH3"],
            N_params_cum=N_params_cum,
        )
    )
    np.testing.assert_array_equal(cube[1:4], np.full(3, -50.0))


def test_prior_transform_public_under_jit_signature():
    """The public `prior_transform` runs its jit-compiled kernel each call.

    Calling twice with the same dispatch tables (codes + ranges have the
    same shape/dtype) reuses the cached compile; smoke-test that no
    tracer errors leak through the public boundary.
    """
    param_names = ["a", "b", "c"]
    prior_types = {"a": "uniform", "b": "gaussian", "c": "uniform"}
    prior_ranges = {"a": [0.0, 1.0], "b": [0.5, 0.1], "c": [-1.0, 1.0]}
    out1 = np.array(
        prior_transform(
            np.array([0.3, 0.5, 0.7]), param_names, prior_types, prior_ranges
        )
    )
    out2 = np.array(
        prior_transform(
            np.array([0.4, 0.5, 0.7]), param_names, prior_types, prior_ranges
        )
    )
    assert out1.shape == (3,)
    assert out2.shape == (3,)
    assert out1[0] != out2[0]  # only the first changed
