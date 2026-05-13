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
    dict(a1=0.5, a2=0.5, log_P1=-3.0, log_P2=-1.0, log_P3=-0.5, T_set=300.0,
         P_set=10.0),
    dict(a1=1.0, a2=0.3, log_P1=-2.0, log_P2=-0.5, log_P3=0.0, T_set=1500.0,
         P_set=10.0),
    dict(a1=0.2, a2=1.5, log_P1=-5.0, log_P2=-3.0, log_P3=-1.0, T_set=800.0,
         P_set=10.0),
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
