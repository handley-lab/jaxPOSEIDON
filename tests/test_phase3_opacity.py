"""Phase 3 opacity-preprocessing tests against POSEIDON."""

import numpy as np
import pytest

from jaxposeidon import _opacity_precompute as op


# ---------------------------------------------------------------------------
# Grid-index helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("value,grid_start,grid_end,N_grid", [
    (5.0, 0.0, 10.0, 11),     # mid grid
    (-1.0, 0.0, 10.0, 11),    # below start
    (11.0, 0.0, 10.0, 11),    # above end
    (0.0, 0.0, 10.0, 11),     # at start
    (10.0, 0.0, 10.0, 11),    # at end
    (0.49, 0.0, 1.0, 101),    # round-down boundary
    (0.5, 0.0, 1.0, 101),     # exactly at .5 → round down
    (0.6, 0.0, 1.0, 101),     # round up
])
def test_closest_index_matches_poseidon(value, grid_start, grid_end, N_grid):
    from POSEIDON.utility import closest_index as p_ci
    assert op.closest_index(value, grid_start, grid_end, N_grid) == \
           p_ci(value, grid_start, grid_end, N_grid)


@pytest.mark.parametrize("value,grid_start,grid_end,N_grid", [
    (5.0, 0.0, 10.0, 11),
    (-1.0, 0.0, 10.0, 11),
    (11.0, 0.0, 10.0, 11),
    (0.0, 0.0, 10.0, 11),
    (10.0, 0.0, 10.0, 11),
])
def test_prior_index_V2_matches_poseidon(value, grid_start, grid_end, N_grid):
    from POSEIDON.utility import prior_index_V2 as p_pi2
    assert op.prior_index_V2(value, grid_start, grid_end, N_grid) == \
           p_pi2(value, grid_start, grid_end, N_grid)


def test_prior_index_matches_poseidon():
    from POSEIDON.utility import prior_index as p_pi
    grid = np.array([100., 150., 200., 300., 500., 1000., 2000., 3500.])
    for v in (50., 100., 175., 250., 999., 2500., 4000.):
        assert op.prior_index(v, grid, 0) == p_pi(v, grid, 0), f"failed at v={v}"


# ---------------------------------------------------------------------------
# T_interpolation_init
# ---------------------------------------------------------------------------
def test_T_interpolation_init_matches_poseidon():
    from POSEIDON.absorption import T_interpolation_init as p_ti
    T_grid = np.array([100., 200., 500., 1000., 2000., 3500.])
    T_fine = np.arange(80., 3700., 50.0)
    N_T_fine = len(T_fine)
    y_ours = np.zeros(N_T_fine, dtype=np.int64)
    y_theirs = np.zeros(N_T_fine, dtype=np.int64)
    w_ours = op.T_interpolation_init(N_T_fine, T_grid, T_fine, y_ours)
    w_theirs = p_ti(N_T_fine, T_grid, T_fine, y_theirs)
    np.testing.assert_array_equal(y_ours, y_theirs)
    np.testing.assert_allclose(w_ours, w_theirs, atol=0)


# ---------------------------------------------------------------------------
# wl_initialise_cia
# ---------------------------------------------------------------------------
def test_wl_initialise_cia_sample_matches_poseidon():
    from POSEIDON.absorption import wl_initialise_cia as p_init
    rng = np.random.default_rng(0)
    N_T, N_wl = 8, 50
    nu_cia = np.linspace(1.0, 5.0, 80)
    nu_model = np.linspace(1.5, 4.5, N_wl)
    log_cia = rng.standard_normal((N_T, len(nu_cia)))
    ours = op.wl_initialise_cia(N_T, N_wl, log_cia, nu_model, nu_cia,
                                len(nu_model), wl_interp="sample")
    theirs = p_init(N_T, N_wl, log_cia, nu_model, nu_cia, len(nu_model),
                    wl_interp="sample")
    np.testing.assert_array_equal(ours, theirs)


def test_wl_initialise_cia_linear_matches_poseidon():
    from POSEIDON.absorption import wl_initialise_cia as p_init
    rng = np.random.default_rng(1)
    N_T, N_wl = 8, 50
    nu_cia = np.linspace(1.0, 5.0, 80)
    nu_model = np.linspace(1.5, 4.5, N_wl)
    log_cia = rng.standard_normal((N_T, len(nu_cia)))
    ours = op.wl_initialise_cia(N_T, N_wl, log_cia, nu_model, nu_cia,
                                len(nu_model), wl_interp="linear")
    theirs = p_init(N_T, N_wl, log_cia, nu_model, nu_cia, len(nu_model),
                    wl_interp="linear")
    np.testing.assert_array_equal(ours, theirs)


# ---------------------------------------------------------------------------
# P_interpolate_wl_initialise_sigma
# ---------------------------------------------------------------------------
def _set_up_sigma_grids(N_P_fine=5, N_T=4, N_P=8, N_wl=20, N_nu_opac=100,
                       wl_interp="sample"):
    rng = np.random.default_rng(2)
    nu_opac = np.linspace(2.0, 10.0, N_nu_opac)
    log_sigma = rng.standard_normal((N_P, N_T, N_nu_opac)) - 20.0
    nu_model = np.linspace(3.0, 9.0, N_wl)
    # x, b1, b2 mimic the log-P interpolation prep that POSEIDON's
    # caller would precompute (using prior_index in log_P).
    # For the test we mix in-range, below-grid, above-grid sentinels:
    x = np.array([1, -1, 3, -2, 5], dtype=np.int64)
    b1 = rng.uniform(0.2, 0.8, size=N_P_fine)
    b2 = 1.0 - b1
    return dict(
        N_P_fine=N_P_fine, N_T=N_T, N_P=N_P, N_wl=N_wl, log_sigma=log_sigma,
        x=x, nu_model=nu_model, b1=b1, b2=b2, nu_opac=nu_opac, N_nu=N_wl,
        wl_interp=wl_interp,
    )


def test_P_interpolate_wl_initialise_sigma_sample_matches_poseidon():
    from POSEIDON.absorption import P_interpolate_wl_initialise_sigma as p_init
    cfg = _set_up_sigma_grids(wl_interp="sample")
    ours = op.P_interpolate_wl_initialise_sigma(**cfg)
    theirs = p_init(
        cfg["N_P_fine"], cfg["N_T"], cfg["N_P"], cfg["N_wl"], cfg["log_sigma"],
        cfg["x"], cfg["nu_model"], cfg["b1"], cfg["b2"], cfg["nu_opac"],
        cfg["N_nu"], wl_interp="sample",
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_P_interpolate_wl_initialise_sigma_linear_matches_poseidon():
    from POSEIDON.absorption import P_interpolate_wl_initialise_sigma as p_init
    cfg = _set_up_sigma_grids(wl_interp="linear")
    ours = op.P_interpolate_wl_initialise_sigma(**cfg)
    theirs = p_init(
        cfg["N_P_fine"], cfg["N_T"], cfg["N_P"], cfg["N_wl"], cfg["log_sigma"],
        cfg["x"], cfg["nu_model"], cfg["b1"], cfg["b2"], cfg["nu_opac"],
        cfg["N_nu"], wl_interp="linear",
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# T_interpolate_sigma / T_interpolate_cia
# ---------------------------------------------------------------------------
def test_T_interpolate_sigma_matches_poseidon():
    from POSEIDON.absorption import T_interpolate_sigma as p_ts
    rng = np.random.default_rng(3)
    N_P_fine, N_T_fine, N_T, N_wl = 4, 6, 5, 20
    sigma_pre_inp = np.abs(rng.standard_normal((N_P_fine, N_T, N_wl))) * 1e-20
    T_grid = np.array([100., 250., 500., 1000., 2000.])
    T_fine = np.array([80., 150., 300., 700., 1500., 3500.])
    y = np.zeros(N_T_fine, dtype=np.int64)
    w_T = op.T_interpolation_init(N_T_fine, T_grid, T_fine, y)
    ours = op.T_interpolate_sigma(N_P_fine, N_T_fine, N_T, N_wl,
                                   sigma_pre_inp, T_grid, T_fine, y, w_T)
    theirs = p_ts(N_P_fine, N_T_fine, N_T, N_wl, sigma_pre_inp, T_grid,
                  T_fine, y, w_T)
    np.testing.assert_allclose(ours, theirs, atol=0)


def test_wl_initialise_cia_linear_at_upper_boundary():
    """Linear interp must not crash at nu_model >= nu_cia[-1] (z=N-1 → opacity 0)."""
    N_T, N_wl = 4, 10
    nu_cia = np.linspace(1.0, 5.0, 40)
    # Push model points across and past the upper edge of nu_cia.
    nu_model = np.linspace(4.5, 6.0, N_wl)
    log_cia = np.random.default_rng(0).standard_normal((N_T, len(nu_cia)))
    out = op.wl_initialise_cia(N_T, N_wl, log_cia, nu_model, nu_cia,
                                N_wl, wl_interp="linear")
    # The out-of-range points must produce zeros, not crash.
    assert np.isfinite(out).all()
    # POSEIDON parity (where it also doesn't crash):
    from POSEIDON.absorption import wl_initialise_cia as p_init
    theirs = p_init(N_T, N_wl, log_cia, nu_model, nu_cia, N_wl,
                    wl_interp="linear")
    np.testing.assert_array_equal(out, theirs)


def test_P_interpolate_wl_initialise_sigma_linear_at_upper_boundary():
    """Linear interp must not crash at nu_model >= nu_opac[-1]."""
    from POSEIDON.absorption import P_interpolate_wl_initialise_sigma as p_init
    rng = np.random.default_rng(5)
    N_P_fine, N_T, N_P, N_wl, N_nu_opac = 3, 3, 5, 8, 30
    nu_opac = np.linspace(2.0, 10.0, N_nu_opac)
    nu_model = np.linspace(9.0, 12.0, N_wl)  # pushes past upper edge
    log_sigma = rng.standard_normal((N_P, N_T, N_nu_opac)) - 20.0
    x = np.array([0, 1, -1], dtype=np.int64)
    b1 = rng.uniform(0.2, 0.8, size=N_P_fine)
    b2 = 1.0 - b1
    ours = op.P_interpolate_wl_initialise_sigma(
        N_P_fine, N_T, N_P, N_wl, log_sigma, x, nu_model, b1, b2,
        nu_opac, N_wl, wl_interp="linear",
    )
    theirs = p_init(
        N_P_fine, N_T, N_P, N_wl, log_sigma, x, nu_model, b1, b2,
        nu_opac, N_wl, wl_interp="linear",
    )
    assert np.isfinite(ours).all()
    np.testing.assert_array_equal(ours, theirs)


def test_T_interpolate_cia_matches_poseidon():
    from POSEIDON.absorption import T_interpolate_cia as p_tc
    rng = np.random.default_rng(4)
    N_T_fine, N_T_cia, N_wl = 6, 5, 20
    cia_pre_inp = np.abs(rng.standard_normal((N_T_cia, N_wl))) * 1e-50
    T_grid_cia = np.array([200., 500., 1000., 2000., 3500.])
    T_fine = np.array([180., 300., 700., 1500., 2500., 4000.])
    y = np.zeros(N_T_fine, dtype=np.int64)
    w_T = op.T_interpolation_init(N_T_fine, T_grid_cia, T_fine, y)
    ours = op.T_interpolate_cia(N_T_fine, N_T_cia, N_wl, cia_pre_inp,
                                 T_grid_cia, T_fine, y, w_T)
    theirs = p_tc(N_T_fine, N_T_cia, N_wl, cia_pre_inp, T_grid_cia,
                  T_fine, y, w_T)
    np.testing.assert_allclose(ours, theirs, atol=0)
