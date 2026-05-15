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
        [rng.uniform(-3.0, 3.0, 50), rng.uniform(0.0, 1.0, 50)], axis=-1
    ).astype(np.float64)
    ref = RegularGridInterpolator(
        (ax0, ax1), values, method="linear", bounds_error=False, fill_value=None
    )(qp)
    ours = np.array(regular_grid_interp_linear((ax0, ax1), values, qp))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_regular_grid_interp_linear_3d_with_boundary_clip():
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
