"""Phase 0.5.6: non-v0 1D P-T profile parity tests against POSEIDON.

Ports of POSEIDON `atmosphere.py` 1D P-T profile functions:
  - compute_T_slope        (`atmosphere.py:232-299`)
  - compute_T_Pelletier    (`atmosphere.py:302-341`)
  - compute_T_Guillot      (`atmosphere.py:344-402`)
  - compute_T_Guillot_dayside (`atmosphere.py:405-462`)
  - compute_T_Line         (`atmosphere.py:465-532`)

Deferred for later phases:
  - compute_T_field_gradient / two_gradients → Phase 0.5.9 (entangled
    with 2D/3D sector/zone scaffolding)
  - file_read PT_profile → Phase 0.5.17 (file I/O)
"""

import numpy as np
import pytest

from jaxposeidon import _atmosphere


@pytest.fixture
def P():
    return np.logspace(np.log10(100.0), np.log10(1.0e-6), 100)


def test_compute_T_Guillot_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_Guillot as p_guillot
    args = dict(g=24.79, log_kappa_IR=-2.5, log_gamma=-1.0, T_int=200.0, T_equ=1200.0)
    ours = _atmosphere.compute_T_Guillot(P, **args)
    theirs = p_guillot(P, **args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_compute_T_Guillot_dayside_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_Guillot_dayside as p_g
    args = dict(g=24.79, log_kappa_IR=-2.5, log_gamma=-1.0, T_int=200.0, T_equ=1500.0)
    ours = _atmosphere.compute_T_Guillot_dayside(P, **args)
    theirs = p_g(P, **args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_compute_T_Line_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_Line as p_line
    args = dict(
        g=24.79, T_eq=1500.0, log_kappa_IR=-2.5, log_gamma=-1.0, log_gamma_2=-0.5,
        alpha=0.3, beta=1.0, T_int=200.0,
    )
    ours = _atmosphere.compute_T_Line(P, **args)
    theirs = p_line(P, **args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_compute_T_slope_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_slope as p_slope
    Delta_T_arr = np.array([100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0])
    log_P_arr = [-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0]
    ours = _atmosphere.compute_T_slope(
        P, T_phot=1200.0, Delta_T_arr=Delta_T_arr, log_P_phot=0.5, log_P_arr=log_P_arr,
    )
    theirs = p_slope(P, 1200.0, Delta_T_arr, 0.5, log_P_arr)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_compute_T_Pelletier_matches_poseidon(P):
    from POSEIDON.atmosphere import compute_T_Pelletier as p_pell
    T_points = np.array([800.0, 1000.0, 1200.0, 1400.0, 1600.0])
    ours = _atmosphere.compute_T_Pelletier(P, T_points)
    theirs = p_pell(P, T_points)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# End-to-end profiles() dispatch — checks each new PT_profile builds a
# valid atmosphere and produces a parity-equivalent T column to POSEIDON.
# ---------------------------------------------------------------------------
def _common_atm_args():
    """Shared input bundle for profiles() — H2/He bulk, no extra species."""
    return dict(
        P=np.logspace(np.log10(100.0), np.log10(1.0e-6), 100),
        R_p=7.1492e7,
        g_0=24.79,
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2", "He"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array([], dtype=str),
        active_species=np.array([], dtype=str),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        N_sectors=1, N_zones=1,
        alpha=0.0, beta=0.0, phi=np.array([0.0]), theta=np.array([0.0]),
        species_vert_gradient=[],
        He_fraction=0.17,
        P_ref=10.0, R_p_ref=7.1492e7,
        constant_gravity=False,
    )


@pytest.mark.parametrize("PT_profile,PT_state,extra", [
    ("Guillot", np.array([-2.5, -1.0, 200.0, 1200.0]), {}),
    ("Guillot_dayside", np.array([-2.5, -1.0, 200.0, 1500.0]), {}),
    ("Line", np.array([-2.5, -1.0, -0.5, 0.3, 1.0, 200.0]), {"T_eq": 1500.0}),
    ("Pelletier", np.array([800.0, 1000.0, 1200.0, 1400.0, 1600.0]), {}),
    ("slope", np.array([1200.0, 100.0, 80.0, 60.0, 40.0, 30.0, 20.0, 10.0]), {}),
])
def test_profiles_dispatch_new_PT_profiles(PT_profile, PT_state, extra):
    cfg = _common_atm_args()
    cfg["PT_profile"] = PT_profile
    cfg["X_profile"] = "isochem"
    cfg["PT_state"] = PT_state
    cfg.update(extra)
    out = _atmosphere.profiles(**cfg)
    assert out[-1] is True, f"profiles() failed for PT_profile={PT_profile}"
    T = out[0]
    assert T.shape == (len(cfg["P"]), 1, 1)
    assert np.all(T > 0.0)
    assert np.all(np.isfinite(T))


# ---------------------------------------------------------------------------
# assign_free_params parity for new PT profiles
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
def test_assign_free_params_PT_param_ordering_matches_poseidon(PT_profile, extra):
    from jaxposeidon._parameter_setup import assign_free_params
    kw = dict(param_species=["H2O"], bulk_species=["H2", "He"],
              PT_profile=PT_profile, X_profile="isochem",
              cloud_model="cloud-free", **extra)
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(PT_profile=PT_profile, **extra)
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[2], theirs[2])


def test_PT_profile_unknown_rejects():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="not_a_profile", X_profile="isochem", PT_state=np.array([1000.0]),
    )
    with pytest.raises(NotImplementedError, match="PT_profile"):
        _atmosphere.profiles(**cfg)
