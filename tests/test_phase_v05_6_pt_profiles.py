"""Phase 0.5.6: non-v0 1D P-T profile parity tests against POSEIDON.

Ports of POSEIDON `atmosphere.py` 1D P-T profile functions:
  - compute_T_slope         (`atmosphere.py:232-299`)
  - compute_T_Pelletier     (`atmosphere.py:302-341`)
  - compute_T_Guillot       (`atmosphere.py:344-402`)
  - compute_T_Guillot_dayside (`atmosphere.py:405-462`)
  - compute_T_Line          (`atmosphere.py:465-532`)

Plus profiles() end-to-end dispatch parity, including:
  - slope's smoothing-width branch
  - Pelletier with PT_penalty=True
  - Line's T_eq requirement
  - matching assign_free_params PT_param ordering + N_params_cumulative

Deferred for later phases:
  - compute_T_field_gradient / two_gradients → Phase 0.5.9
  - file_read PT_profile → Phase 0.5.17
"""

import numpy as np
import pytest

from jaxposeidon import _atmosphere


@pytest.fixture
def P():
    return np.logspace(np.log10(100.0), np.log10(1.0e-6), 100)


# ---------------------------------------------------------------------------
# Direct PT-function parity (bit-exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("log_kappa_IR,log_gamma,T_int,T_equ", [
    (-2.5, -1.0, 200.0, 1200.0),
    (-1.0, 0.0, 100.0, 800.0),
    (-3.5, -1.5, 500.0, 1800.0),
])
def test_compute_T_Guillot_matches_poseidon(P, log_kappa_IR, log_gamma, T_int, T_equ):
    from POSEIDON.atmosphere import compute_T_Guillot as p_guillot
    args = dict(g=24.79, log_kappa_IR=log_kappa_IR, log_gamma=log_gamma,
                T_int=T_int, T_equ=T_equ)
    np.testing.assert_array_equal(
        _atmosphere.compute_T_Guillot(P, **args), p_guillot(P, **args)
    )


@pytest.mark.parametrize("log_kappa_IR,log_gamma,T_int,T_equ", [
    (-2.5, -1.0, 200.0, 1500.0),
    (-1.0, 0.0, 100.0, 1000.0),
])
def test_compute_T_Guillot_dayside_matches_poseidon(P, log_kappa_IR, log_gamma, T_int, T_equ):
    from POSEIDON.atmosphere import compute_T_Guillot_dayside as p_g
    args = dict(g=24.79, log_kappa_IR=log_kappa_IR, log_gamma=log_gamma,
                T_int=T_int, T_equ=T_equ)
    np.testing.assert_array_equal(
        _atmosphere.compute_T_Guillot_dayside(P, **args), p_g(P, **args)
    )


@pytest.mark.parametrize("alpha,beta", [(0.3, 1.0), (0.0, 1.0), (1.0, 0.5), (0.5, 1.5)])
def test_compute_T_Line_matches_poseidon(P, alpha, beta):
    from POSEIDON.atmosphere import compute_T_Line as p_line
    args = dict(
        g=24.79, T_eq=1500.0, log_kappa_IR=-2.5, log_gamma=-1.0, log_gamma_2=-0.5,
        alpha=alpha, beta=beta, T_int=200.0,
    )
    np.testing.assert_array_equal(
        _atmosphere.compute_T_Line(P, **args), p_line(P, **args)
    )


def test_compute_T_slope_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_slope as p_slope
    Delta_T_arr = np.array([100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0])
    log_P_arr = [-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0]
    np.testing.assert_array_equal(
        _atmosphere.compute_T_slope(P, 1200.0, Delta_T_arr, 0.5, log_P_arr),
        p_slope(P, 1200.0, Delta_T_arr, 0.5, log_P_arr),
    )


@pytest.mark.parametrize("n_knots", [3, 5, 7])
def test_compute_T_Pelletier_matches_poseidon(P, n_knots):
    from POSEIDON.atmosphere import compute_T_Pelletier as p_pell
    T_points = np.linspace(800.0, 1600.0, n_knots)
    np.testing.assert_array_equal(
        _atmosphere.compute_T_Pelletier(P, T_points),
        p_pell(P, T_points),
    )


# ---------------------------------------------------------------------------
# End-to-end profiles() POSEIDON parity
# ---------------------------------------------------------------------------
def _common_atm_args(N_layers=50):
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    return dict(
        P=P, R_p=7.1492e7, g_0=24.79,
        P_ref=10.0, R_p_ref=7.1492e7,
        N_sectors=1, N_zones=1,
        alpha=0.0, beta=0.0,
        phi=np.array([0.0]), theta=np.array([0.0]),
        species_vert_gradient=np.array([], dtype=str),
        He_fraction=0.17,
        P_param_set=1.0e-6,
        log_P_slope_phot=0.5,
        log_P_slope_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2", "He"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array([], dtype=str),
        active_species=np.array([], dtype=str),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=False,
    )


def _profiles_assert_match(cfg):
    from POSEIDON.atmosphere import profiles as p_profiles
    ours = _atmosphere.profiles(**cfg)
    theirs = p_profiles(
        cfg["P"], cfg["R_p"], cfg["g_0"], cfg["PT_profile"], cfg["X_profile"],
        cfg["PT_state"], cfg["P_ref"], cfg["R_p_ref"], cfg["log_X_state"],
        cfg["included_species"], cfg["bulk_species"], cfg["param_species"],
        cfg["active_species"], cfg["CIA_pairs"], cfg["ff_pairs"], cfg["bf_species"],
        cfg["N_sectors"], cfg["N_zones"], cfg["alpha"], cfg["beta"],
        cfg["phi"], cfg["theta"], cfg["species_vert_gradient"], cfg["He_fraction"],
        cfg.get("T_input", None), cfg.get("X_input", None),
        cfg.get("P_param_set", 1.0e-6),
        cfg.get("log_P_slope_phot", 0.5),
        list(cfg.get("log_P_slope_arr", (-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0))),
        cfg.get("Na_K_fixed_ratio", False),
        cfg.get("constant_gravity", False),
        cfg.get("chemistry_grid", None),
        cfg.get("PT_penalty", False),
        cfg.get("T_eq", None),
        cfg.get("mu_back", None),
        cfg.get("disable_atmosphere", False),
    )
    assert len(ours) == len(theirs) == 13
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)):
            assert a == b, f"physical-flag mismatch at index {i}: {a} vs {b}"
        else:
            np.testing.assert_array_equal(
                a, b, err_msg=f"profiles() output {i} differs"
            )


def test_profiles_Guillot_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="Guillot", X_profile="isochem",
        PT_state=np.array([-2.5, -1.0, 200.0, 1200.0]),
    )
    _profiles_assert_match(cfg)


def test_profiles_Guillot_dayside_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="Guillot_dayside", X_profile="isochem",
        PT_state=np.array([-2.5, -1.0, 200.0, 1500.0]),
    )
    _profiles_assert_match(cfg)


def test_profiles_Line_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="Line", X_profile="isochem",
        PT_state=np.array([-2.5, -1.0, -0.5, 0.3, 1.0, 200.0]),
        T_eq=1500.0,
    )
    _profiles_assert_match(cfg)


@pytest.mark.parametrize("N_layers", [50, 100, 75])
def test_profiles_slope_matches_poseidon(N_layers):
    """Slope profile through the smoothing branch — sensitive to smooth_width."""
    cfg = _common_atm_args(N_layers=N_layers)
    cfg.update(
        PT_profile="slope", X_profile="isochem",
        PT_state=np.array([1200.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0]),
    )
    _profiles_assert_match(cfg)


@pytest.mark.parametrize("n_knots,PT_penalty", [(3, False), (5, False), (7, True)])
def test_profiles_Pelletier_matches_poseidon(n_knots, PT_penalty):
    cfg = _common_atm_args()
    T_points = list(np.linspace(800.0, 1600.0, n_knots))
    PT_state = np.array(T_points + ([0.1] if PT_penalty else []))
    cfg.update(
        PT_profile="Pelletier", X_profile="isochem",
        PT_state=PT_state, PT_penalty=PT_penalty,
    )
    _profiles_assert_match(cfg)


# ---------------------------------------------------------------------------
# assign_free_params parity — full-tuple comparison
# ---------------------------------------------------------------------------
def _poseidon_assign(**overrides):
    from POSEIDON.parameters import assign_free_params as p_assign
    defaults = dict(
        param_species=["H2O"], bulk_species=["H2", "He"],
        object_type="transiting",
        PT_profile="isotherm", X_profile="isochem",
        cloud_model="cloud-free", cloud_type="deck",
        gravity_setting="fixed", mass_setting="fixed",
        stellar_contam=None, offsets_applied=None, error_inflation=None,
        PT_dim=1, X_dim=1, cloud_dim=1,
        TwoD_type=None, TwoD_param_scheme="difference",
        species_EM_gradient=[], species_DN_gradient=[],
        species_vert_gradient=[],
        Atmosphere_dimension=1, opaque_Iceberg=False, surface=False,
        sharp_DN_transition=False, sharp_EM_transition=False,
        reference_parameter="R_p_ref", disable_atmosphere=False,
        aerosol_species=[],
        log_P_slope_arr=[-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0],
        number_P_knots=0, PT_penalty=False,
        high_res_method=None, alpha_high_res_option="log",
        fix_alpha_high_res=False, fix_W_conv_high_res=False,
        fix_beta_high_res=True, fix_Delta_phi_high_res=True,
        lognormal_logwidth_free=False,
        surface_components=[], surface_model="gray",
        surface_percentage_option="linear",
        thermal=True, reflection=False,
    )
    defaults.update(overrides)
    return p_assign(**defaults)


@pytest.mark.parametrize("PT_profile,extra", [
    ("slope", {}),
    ("Pelletier", {"number_P_knots": 5}),
    ("Pelletier", {"number_P_knots": 7, "PT_penalty": True}),
    ("Guillot", {}),
    ("Guillot_dayside", {}),
    ("Line", {}),
])
def test_assign_free_params_full_tuple_matches_poseidon(PT_profile, extra):
    """Full-tuple parity including N_params_cumulative (used by split_params)."""
    from jaxposeidon._parameter_setup import assign_free_params
    kw = dict(param_species=["H2O"], bulk_species=["H2", "He"],
              PT_profile=PT_profile, X_profile="isochem",
              cloud_model="cloud-free", **extra)
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(PT_profile=PT_profile, **extra)
    assert len(ours) == len(theirs)
    for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
        np.testing.assert_array_equal(a, b, err_msg=f"tuple element {i} differs")


# ---------------------------------------------------------------------------
# Rejection / error-path tests
# ---------------------------------------------------------------------------
def test_PT_profile_unknown_rejects():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="not_a_profile", X_profile="isochem",
        PT_state=np.array([1000.0]),
    )
    with pytest.raises(NotImplementedError, match="PT_profile"):
        _atmosphere.profiles(**cfg)


def test_PT_profile_Line_without_T_eq_rejects():
    """T_eq is required for the Line profile (POSEIDON passes it through `T_eq`)."""
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="Line", X_profile="isochem",
        PT_state=np.array([-2.5, -1.0, -0.5, 0.3, 1.0, 200.0]),
    )
    with pytest.raises(ValueError, match="T_eq"):
        _atmosphere.profiles(**cfg)
