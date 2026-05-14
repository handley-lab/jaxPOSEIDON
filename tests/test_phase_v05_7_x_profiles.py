"""Phase 0.5.7: non-isochem X-profile + dissociation + ghost-bulk + Na_K parity.

Ports of POSEIDON `atmosphere.py`:
  - compute_X_field_gradient       (`atmosphere.py:759-878`)
  - compute_X_field_two_gradients  (`atmosphere.py:882-1013`)
  - Parmentier_dissociation_profile (`atmosphere.py:1017-1052`)
  - compute_X_dissociation          (`atmosphere.py:1056-1166`)
  - compute_X_lever                 (`atmosphere.py:1169-1219`)
  - add_bulk_component H2+H+He dissociation branch (`atmosphere.py:1285-1312`)

Plus profiles() end-to-end parity with X_profile in
{gradient, two-gradients, dissociation, lever, file_read} and Na_K_fixed_ratio
and mu_back (ghost bulk).
"""

import numpy as np
import pytest

from jaxposeidon import _atmosphere


P_GRID = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)


# ---------------------------------------------------------------------------
# Direct X-function parity
# ---------------------------------------------------------------------------
def test_compute_X_field_gradient_matches_poseidon():
    from POSEIDON.atmosphere import compute_X_field_gradient as p_grad
    P = P_GRID
    log_X_state = np.array([[-3.0, 0.0, 0.0, -4.0], [-4.5, 0.0, 0.0, -5.5]])
    param_species = np.array(["H2O", "CH4"])
    species_has_profile = np.array([1, 0], dtype=np.int64)
    phi = np.array([0.0])
    theta = np.array([0.0])
    ours = _atmosphere.compute_X_field_gradient(
        P, log_X_state, 1, 1, param_species, species_has_profile,
        0.0, 0.0, phi, theta,
    )
    theirs = p_grad(
        P, log_X_state, 1, 1, param_species, species_has_profile,
        0.0, 0.0, phi, theta,
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_compute_X_field_two_gradients_matches_poseidon():
    from POSEIDON.atmosphere import compute_X_field_two_gradients as p_two
    P = P_GRID
    log_X_state = np.array([[-3.0, -3.5, 0.0, 0.0, 0.0, 0.0, -1.0, -4.5]])
    param_species = np.array(["H2O"])
    species_has_profile = np.array([1], dtype=np.int64)
    phi = np.array([0.0])
    theta = np.array([0.0])
    ours = _atmosphere.compute_X_field_two_gradients(
        P, log_X_state, 1, 1, param_species, species_has_profile,
        0.0, 0.0, phi, theta,
    )
    theirs = p_two(
        P, log_X_state, 1, 1, param_species, species_has_profile,
        0.0, 0.0, phi, theta,
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_Parmentier_dissociation_profile_matches_poseidon():
    from POSEIDON.atmosphere import Parmentier_dissociation_profile as p_diss
    P = P_GRID
    T = 2500.0 * np.ones_like(P)
    A_0 = 1e-3
    alpha, beta, gamma, A_0_ref = 2.0, 4.83e4, 15.9, 10 ** -3.3
    np.testing.assert_allclose(
        _atmosphere.Parmentier_dissociation_profile(P, T, A_0, alpha, beta, gamma, A_0_ref),
        p_diss(P, T, A_0, alpha, beta, gamma, A_0_ref),
        atol=0, rtol=1e-13,
    )


@pytest.mark.parametrize("species", ["H2O", "TiO", "VO", "H-", "Na", "K"])
def test_compute_X_dissociation_matches_poseidon(species):
    from POSEIDON.atmosphere import compute_X_dissociation as p_diss
    P = P_GRID
    T = 2200.0 * np.ones((len(P), 1, 1))
    log_X_state = np.array([[-3.0, 0.0, 0.0]])
    param_species = np.array([species])
    species_has_profile = np.array([1], dtype=np.int64)
    phi = np.array([0.0])
    theta = np.array([0.0])
    np.testing.assert_allclose(
        _atmosphere.compute_X_dissociation(
            P, T, log_X_state, 1, 1, param_species, species_has_profile,
            0.0, 0.0, phi, theta,
        ),
        p_diss(
            P, T, log_X_state, 1, 1, param_species, species_has_profile,
            0.0, 0.0, phi, theta,
        ),
        atol=0, rtol=1e-13,
    )


def test_compute_X_lever_matches_poseidon():
    from POSEIDON.atmosphere import compute_X_lever as p_lever
    P = P_GRID
    log_X_state = np.array([[-3.0, -1.5, 30.0], [-4.0, -2.0, 60.0]])
    species_has_profile = np.array([1, 1], dtype=np.int64)
    np.testing.assert_array_equal(
        _atmosphere.compute_X_lever(P, log_X_state, species_has_profile, 1, 1),
        p_lever(P, log_X_state, species_has_profile, 1, 1),
    )


# ---------------------------------------------------------------------------
# profiles() POSEIDON parity for new X_profile values
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
        constant_gravity=False,
    )


def _profiles_assert_match(cfg, rtol=0, atol=0):
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
            if rtol == 0 and atol == 0:
                np.testing.assert_array_equal(
                    a, b, err_msg=f"profiles() output {i} differs"
                )
            else:
                np.testing.assert_allclose(
                    a, b, rtol=rtol, atol=atol,
                    err_msg=f"profiles() output {i} differs"
                )


def test_profiles_X_gradient_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="gradient",
        PT_state=np.array([1200.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -4.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        species_vert_gradient=np.array(["H2O"]),
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_X_two_gradients_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="two-gradients",
        PT_state=np.array([1200.0]),
        log_X_state=np.array([[-3.0, -3.5, 0.0, 0.0, 0.0, 0.0, -1.0, -4.5]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        species_vert_gradient=np.array(["H2O"]),
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_X_dissociation_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="dissociation",
        PT_state=np.array([2200.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        species_vert_gradient=np.array(["H2O"]),
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_X_lever_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="lever",
        PT_state=np.array([1200.0]),
        log_X_state=np.array([[-3.0, -1.5, 30.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        species_vert_gradient=np.array(["H2O"]),
    )
    _profiles_assert_match(cfg)


def test_profiles_X_file_read_matches_poseidon():
    """User-supplied X via X_input."""
    cfg = _common_atm_args()
    N = len(cfg["P"])
    X_input = np.zeros((3, N))
    X_input[0] = 0.85  # H2
    X_input[1] = 0.149  # He
    X_input[2] = 0.001  # H2O
    cfg.update(
        PT_profile="isotherm", X_profile="file_read",
        PT_state=np.array([1200.0]),
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        X_input=X_input,
    )
    _profiles_assert_match(cfg)


def test_profiles_PT_file_read_matches_poseidon():
    cfg = _common_atm_args()
    N = len(cfg["P"])
    T_input = np.linspace(800.0, 1500.0, N)
    cfg.update(
        PT_profile="file_read", X_profile="isochem",
        PT_state=np.array([]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        T_input=T_input,
    )
    _profiles_assert_match(cfg)


def test_profiles_PT_gradient_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="gradient", X_profile="isochem",
        PT_state=np.array([1000.0, 200.0, 0.0, 1500.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_PT_two_gradients_matches_poseidon():
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="two-gradients", X_profile="isochem",
        PT_state=np.array([1000.0, 1200.0, 200.0, 100.0, 0.0, 0.0, -1.0, 1500.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    _profiles_assert_match(cfg)


def test_profiles_H2_H_He_dissociation_bulk_matches_poseidon():
    """H2+H+He bulk with Parmentier H2 dissociation."""
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([2500.0]),
        log_X_state=np.array([[-4.0, 0.0, 0.0, -4.0]]),
        included_species=np.array(["H2", "H", "He", "H2O"]),
        bulk_species=np.array(["H2", "H", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_Na_K_fixed_ratio_matches_poseidon():
    """Na_K_fixed_ratio adds K mixing ratio = 0.1 × Na.

    POSEIDON's `K_X_state = [X_param[param_species.index('Na')]...]` requires
    `param_species` to be a list (numpy arrays lack `.index`); we pass a list
    here to mirror POSEIDON's expected input type.
    """
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([1500.0]),
        log_X_state=np.array([[-6.0, 0.0, 0.0, -6.0]]),
        included_species=np.array(["H2", "He", "Na", "K"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=["Na"],
        active_species=np.array(["Na", "K"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        Na_K_fixed_ratio=True,
    )
    _profiles_assert_match(cfg, rtol=1e-13, atol=0)


def test_profiles_ghost_mu_back_matches_poseidon():
    """ghost bulk species uses mu_back as molecular mass."""
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([1200.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["ghost", "H2O"]),
        bulk_species=np.array(["ghost"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        mu_back=2.3,
    )
    _profiles_assert_match(cfg)
