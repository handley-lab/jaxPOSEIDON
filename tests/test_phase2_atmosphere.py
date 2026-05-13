"""Phase 2 tests for jaxposeidon._atmosphere and _geometry against POSEIDON."""

import numpy as np
import pytest

from jaxposeidon import _atmosphere, _geometry


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_atmosphere_regions_1D_matches_poseidon():
    from POSEIDON.geometry import atmosphere_regions as p_regions
    n_s, n_z = _geometry.atmosphere_regions(1)
    p_n_s, p_n_z = p_regions(1, None, 2, 2)
    assert (n_s, n_z) == (p_n_s, p_n_z) == (1, 1)


def test_atmosphere_regions_rejects_higher_dim():
    with pytest.raises(NotImplementedError, match="Atmosphere_dimension=1"):
        _geometry.atmosphere_regions(2, "D-N", 2, 2)
    with pytest.raises(NotImplementedError):
        _geometry.atmosphere_regions(3)


def test_angular_grids_1D_matches_poseidon():
    from POSEIDON.geometry import angular_grids as p_angular
    ours = _geometry.angular_grids(1, None, 2, 2, 0.0, 0.0, False, False)
    theirs = p_angular(1, None, 2, 2, 0.0, 0.0, False, False)
    assert len(ours) == len(theirs) == 6
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


def test_angular_grids_rejects_higher_dim():
    with pytest.raises(NotImplementedError):
        _geometry.angular_grids(2, "D-N", 2, 2, 0.0, 30.0, False, False)


# ---------------------------------------------------------------------------
# P-T profiles
# ---------------------------------------------------------------------------
def test_compute_T_isotherm():
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    T = _atmosphere.compute_T_isotherm(P, 1000.0)
    assert T.shape == (100, 1, 1)
    assert np.all(T == 1000.0)


def test_compute_T_Madhu_matches_poseidon():
    from POSEIDON.atmosphere import compute_T_Madhu as p_madhu
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    # K2-18 b-like params from paper Table A2
    args = dict(a1=0.5, a2=0.5, log_P1=-3.0, log_P2=-1.0, log_P3=-0.5,
                T_set=300.0, P_set=10.0)
    T_ours = _atmosphere.compute_T_Madhu(P, **args)
    T_theirs = p_madhu(P, args["a1"], args["a2"], args["log_P1"],
                       args["log_P2"], args["log_P3"], args["T_set"],
                       args["P_set"])
    np.testing.assert_array_equal(T_ours, T_theirs)


@pytest.mark.parametrize("params", [
    # Layer 3 branch (P_set in deep layer, log_P_set_i >= log_P3)
    dict(a1=0.5, a2=0.5, log_P1=-3.0, log_P2=-1.0, log_P3=-0.5, T_set=300.0,
         P_set=10.0),
    dict(a1=1.0, a2=0.3, log_P1=-2.0, log_P2=-0.5, log_P3=0.0, T_set=1500.0,
         P_set=10.0),
    dict(a1=0.2, a2=1.5, log_P1=-5.0, log_P2=-3.0, log_P3=-1.0, T_set=800.0,
         P_set=10.0),
    # Layer 2 branch: log_P1 < log_P_set_i < log_P3
    dict(a1=0.5, a2=0.5, log_P1=-5.0, log_P2=-3.0, log_P3=-1.0, T_set=300.0,
         P_set=1.0e-2),  # log_P_set ≈ -2, between log_P1=-5 and log_P3=-1
    # Layer 1 branch: log_P_set_i < log_P1
    dict(a1=0.5, a2=0.5, log_P1=-2.0, log_P2=-1.0, log_P3=-0.5, T_set=300.0,
         P_set=1.0e-6),  # log_P_set ≈ -6, below log_P1=-2
])
def test_compute_T_Madhu_param_sweep(params):
    from POSEIDON.atmosphere import compute_T_Madhu as p_madhu
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    T_ours = _atmosphere.compute_T_Madhu(P, **params)
    T_theirs = p_madhu(P, params["a1"], params["a2"], params["log_P1"],
                       params["log_P2"], params["log_P3"], params["T_set"],
                       params["P_set"])
    np.testing.assert_array_equal(T_ours, T_theirs)


def test_gauss_conv_matches_poseidon():
    """Verify our gauss_conv is the same as POSEIDON's (both are scipy bridges)."""
    from POSEIDON.atmosphere import gauss_conv as p_gauss
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((100, 1, 1))
    np.testing.assert_array_equal(_atmosphere.gauss_conv(arr),
                                  p_gauss(arr, sigma=3, axis=0, mode="nearest"))


# ---------------------------------------------------------------------------
# Mixing-ratio construction
# ---------------------------------------------------------------------------
def test_compute_X_isochem_1D_constant_in_all_axes():
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    log_X_state = np.array([[-3.5, 0.0, 0.0, -3.5],   # H2O
                            [-4.0, 0.0, 0.0, -4.0]])  # CH4
    X = _atmosphere.compute_X_isochem_1D(P, log_X_state, 1, 1, ["H2O", "CH4"])
    assert X.shape == (2, 100, 1, 1)
    np.testing.assert_allclose(X[0], 10.0 ** -3.5)
    np.testing.assert_allclose(X[1], 10.0 ** -4.0)


def test_compute_X_isochem_1D_matches_poseidon_field_gradient():
    """v0 isochem path is equivalent to POSEIDON's compute_X_field_gradient
    in the X_dim=1, no-vert-gradient configuration."""
    from POSEIDON.atmosphere import compute_X_field_gradient as p_x_field
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    log_X_state = np.array([[-3.5, 0.0, 0.0, -3.5]])
    species_has_profile = np.zeros(1, dtype=np.int64)  # no vertical gradient
    X_ours = _atmosphere.compute_X_isochem_1D(P, log_X_state, 1, 1, ["H2O"])
    X_theirs = p_x_field(P, log_X_state, 1, 1, ["H2O"], species_has_profile,
                         0.0, 0.0, np.array([0.0]), np.array([0.0]))
    np.testing.assert_array_equal(X_ours, X_theirs)


def test_add_bulk_component_H2_He():
    from POSEIDON.atmosphere import add_bulk_component as p_bulk
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    T = 300.0 * np.ones((100, 1, 1))
    X_param = 0.01 * np.ones((2, 100, 1, 1))  # 1% H2O + 1% CH4
    He_fraction = 0.17
    X_ours = _atmosphere.add_bulk_component(
        P, T, X_param, N_species=4, N_sectors=1, N_zones=1,
        bulk_species=["H2", "He"], He_fraction=He_fraction,
    )
    X_theirs = p_bulk(P, T, X_param, 4, 1, 1, ["H2", "He"], He_fraction)
    np.testing.assert_array_equal(X_ours, X_theirs)
    # H2 and He account for the remainder.
    np.testing.assert_allclose(X_ours.sum(axis=0), 1.0, atol=1e-15)


def test_add_bulk_component_single_species():
    from POSEIDON.atmosphere import add_bulk_component as p_bulk
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    T = 300.0 * np.ones((100, 1, 1))
    X_param = 0.001 * np.ones((1, 100, 1, 1))
    X_ours = _atmosphere.add_bulk_component(P, T, X_param, 2, 1, 1, ["N2"], 0.17)
    X_theirs = p_bulk(P, T, X_param, 2, 1, 1, ["N2"], 0.17)
    np.testing.assert_array_equal(X_ours, X_theirs)


def test_add_bulk_component_H_He():
    """H+He bulk path (no H2)."""
    from POSEIDON.atmosphere import add_bulk_component as p_bulk
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
    T = 300.0 * np.ones((50, 1, 1))
    X_param = 0.001 * np.ones((1, 50, 1, 1))
    He_fraction = 0.17
    X_ours = _atmosphere.add_bulk_component(P, T, X_param, 3, 1, 1, ["H", "He"], He_fraction)
    X_theirs = p_bulk(P, T, X_param, 3, 1, 1, ["H", "He"], He_fraction)
    np.testing.assert_array_equal(X_ours, X_theirs)


def test_add_bulk_component_H2_H_He_rejected():
    """H2+H+He dissociation bulk is v0-deferred (raises NotImplementedError)."""
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 20)
    T = 300.0 * np.ones((20, 1, 1))
    X_param = 0.001 * np.ones((1, 20, 1, 1))
    with pytest.raises(NotImplementedError, match="dissociation"):
        _atmosphere.add_bulk_component(P, T, X_param, 4, 1, 1,
                                        ["H2", "H", "He"], 0.17)


# ---------------------------------------------------------------------------
# Mean molecular mass
# ---------------------------------------------------------------------------
def test_compute_mean_mol_mass_matches_poseidon():
    from POSEIDON.atmosphere import compute_mean_mol_mass as p_mu
    from POSEIDON.species_data import masses
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    species = ["H2", "He", "H2O"]
    masses_all = np.array([masses[s] for s in species])
    X = np.zeros((3, 100, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    np.testing.assert_array_equal(
        _atmosphere.compute_mean_mol_mass(P, X, 3, 1, 1, masses_all),
        p_mu(P, X, 3, 1, 1, masses_all),
    )


# ---------------------------------------------------------------------------
# Radial profiles
# ---------------------------------------------------------------------------
def _setup_simple_atm():
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
    T = 1000.0 * np.ones((50, 1, 1))
    mu = 2.33 * 1.66053906660e-27 * np.ones((50, 1, 1))
    return P, T, mu


def test_radial_profiles_matches_poseidon():
    from POSEIDON.atmosphere import radial_profiles as p_rp
    P, T, mu = _setup_simple_atm()
    R_p = 7.1492e7
    g_0 = 24.79
    P_ref = 100.0
    R_p_ref = R_p
    ours = _atmosphere.radial_profiles(P, T, g_0, R_p, P_ref, R_p_ref, mu, 1, 1)
    theirs = p_rp(P, T, g_0, R_p, P_ref, R_p_ref, mu, 1, 1)
    assert len(ours) == len(theirs) == 5
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


def test_radial_profiles_constant_g_matches_poseidon():
    from POSEIDON.atmosphere import radial_profiles_constant_g as p_rpcg
    P, T, mu = _setup_simple_atm()
    g_0 = 24.79
    P_ref = 100.0
    R_p_ref = 7.1492e7
    ours = _atmosphere.radial_profiles_constant_g(
        P, T, g_0, P_ref, R_p_ref, mu, 1, 1)
    theirs = p_rpcg(P, T, g_0, P_ref, R_p_ref, mu, 1, 1)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("R_p,g_0,P_ref", [
    (7.1492e7, 24.79, 100.0),    # Jupiter-like (test_TRIDENT default)
    (1.665e7, 12.4,  10.0),       # K2-18b
    (6.378e6, 9.81,  1.0),        # Earth-like
])
def test_radial_profiles_param_sweep(R_p, g_0, P_ref):
    from POSEIDON.atmosphere import radial_profiles as p_rp
    P, T, mu = _setup_simple_atm()
    ours = _atmosphere.radial_profiles(P, T, g_0, R_p, P_ref, R_p, mu, 1, 1)
    theirs = p_rp(P, T, g_0, R_p, P_ref, R_p, mu, 1, 1)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# mixing_ratio_categories
# ---------------------------------------------------------------------------
def test_mixing_ratio_categories_matches_poseidon():
    from POSEIDON.atmosphere import mixing_ratio_categories as p_mrc
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 30)
    included = np.array(["H2", "He", "H2O", "CH4", "CO2"])
    active = np.array(["H2O", "CH4", "CO2"])
    CIA = np.array(["H2-H2", "H2-He"])
    ff = np.array([], dtype=str)
    bf = np.array([], dtype=str)
    rng = np.random.default_rng(0)
    X = rng.uniform(1e-6, 1e-1, size=(5, 30, 1, 1))
    ours = _atmosphere.mixing_ratio_categories(P, X, 1, 1, included, active, CIA, ff, bf)
    theirs = p_mrc(P, X, 1, 1, included, active, CIA, ff, bf)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# profiles() full v0 dispatcher
# ---------------------------------------------------------------------------
def _common_atm_args(P_ref=100.0, R_p=7.1492e7, g_0=24.79):
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
    return dict(
        P=P, R_p=R_p, g_0=g_0,
        P_ref=P_ref, R_p_ref=R_p,
        N_sectors=1, N_zones=1,
        alpha=0.0, beta=0.0,
        phi=np.array([0.0]), theta=np.array([0.0]),
        species_vert_gradient=np.array([], dtype=str),
        He_fraction=0.17,
        P_param_set=1.0e-6,
        log_P_slope_phot=0.5,
        log_P_slope_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
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
    # T, n, r, r_up, r_low, dr, mu, X, X_active, X_CIA, X_ff, X_bf, physical
    for i, (a, b) in enumerate(zip(ours, theirs)):
        if isinstance(a, bool) or isinstance(b, (bool, np.bool_)):
            assert a == b, f"physical-flag mismatch at index {i}: {a} vs {b}"
        else:
            np.testing.assert_array_equal(a, b,
                err_msg=f"profiles() output {i} differs")


def test_profiles_isotherm_constant_gravity_H2_only():
    """Canonical Rayleigh oracle config: isotherm, single H2 bulk, constant g."""
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([1000.0]),
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2"]),
        bulk_species=np.array(["H2"]),
        param_species=np.array([], dtype=str),
        active_species=np.array([], dtype=str),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=True,
    )
    _profiles_assert_match(cfg)


def test_profiles_isotherm_inverse_sq_H2_He_with_H2O():
    """Inverse-sq gravity, H2/He bulk, single H2O param species."""
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([300.0]),
        log_X_state=np.array([[-3.5, 0.0, 0.0, -3.5]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array(["H2-H2", "H2-He"]),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=False,
    )
    _profiles_assert_match(cfg)


def test_profiles_Madhu_constant_g_k2_18b():
    """K2-18 b–like Madhu PT + isochem with multiple species, constant g."""
    cfg = _common_atm_args(P_ref=10.0, R_p=1.665e7, g_0=12.4)
    cfg.update(
        PT_profile="Madhu", X_profile="isochem",
        PT_state=np.array([0.5, 0.5, -3.0, -1.0, -0.5, 250.0]),
        log_X_state=np.array([
            [-3.0, 0.0, 0.0, -3.0],
            [-2.0, 0.0, 0.0, -2.0],
        ]),
        included_species=np.array(["H2", "He", "CH4", "CO2"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["CH4", "CO2"]),
        active_species=np.array(["CH4", "CO2"]),
        CIA_pairs=np.array(["H2-H2", "H2-He"]),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=True,
    )
    _profiles_assert_match(cfg)


def test_profiles_Madhu_inverse_sq_k2_18b():
    """K2-18 b–like Madhu + isochem, inverse-sq gravity."""
    cfg = _common_atm_args(P_ref=10.0, R_p=1.665e7, g_0=12.4)
    cfg.update(
        PT_profile="Madhu", X_profile="isochem",
        PT_state=np.array([0.5, 0.5, -3.0, -1.0, -0.5, 250.0]),
        log_X_state=np.array([[-3.5, 0.0, 0.0, -3.5]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array(["H2-H2", "H2-He"]),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=False,
    )
    _profiles_assert_match(cfg)


def test_profiles_madhu_rejects_invalid_pressure_ordering():
    """Madhu profile with log_P3 < log_P2 must return physical=False."""
    from POSEIDON.atmosphere import profiles as p_profiles
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="Madhu", X_profile="isochem",
        PT_state=np.array([0.5, 0.5, -3.0, -0.5, -1.0, 250.0]),  # log_P3<log_P2
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2"]),
        bulk_species=np.array(["H2"]),
        param_species=np.array([], dtype=str),
        active_species=np.array([], dtype=str),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
        constant_gravity=True,
    )
    ours = _atmosphere.profiles(**cfg)
    assert ours[-1] is False
    # POSEIDON also returns physical=False
    theirs = p_profiles(
        cfg["P"], cfg["R_p"], cfg["g_0"], cfg["PT_profile"], cfg["X_profile"],
        cfg["PT_state"], cfg["P_ref"], cfg["R_p_ref"], cfg["log_X_state"],
        cfg["included_species"], cfg["bulk_species"], cfg["param_species"],
        cfg["active_species"], cfg["CIA_pairs"], cfg["ff_pairs"], cfg["bf_species"],
        1, 1, 0.0, 0.0, np.array([0.0]), np.array([0.0]),
        np.array([], dtype=str), 0.17,
    )
    assert theirs[-1] is False


@pytest.mark.parametrize("kwargs,err_substring", [
    (dict(PT_profile="gradient"), "PT_profile"),
    (dict(PT_profile="Guillot"), "PT_profile"),
    (dict(X_profile="gradient"), "X_profile"),
    (dict(X_profile="chem_eq"), "X_profile"),
    (dict(disable_atmosphere=True), "disable_atmosphere"),
    (dict(N_sectors=2), "N_sectors=N_zones=1"),
    (dict(species_vert_gradient=np.array(["H2O"])), "vertical gradient"),
    (dict(Na_K_fixed_ratio=True), "Na_K_fixed_ratio"),
    (dict(mu_back=2.3), "ghost-bulk"),
])
def test_profiles_rejects_non_v0(kwargs, err_substring):
    cfg = _common_atm_args()
    cfg.update(
        PT_profile="isotherm", X_profile="isochem",
        PT_state=np.array([1000.0]),
        log_X_state=np.zeros((0, 4)),
        included_species=np.array(["H2"]),
        bulk_species=np.array(["H2"]),
        param_species=np.array([], dtype=str),
        active_species=np.array([], dtype=str),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    cfg.update(kwargs)
    with pytest.raises(NotImplementedError, match=err_substring):
        _atmosphere.profiles(**cfg)
