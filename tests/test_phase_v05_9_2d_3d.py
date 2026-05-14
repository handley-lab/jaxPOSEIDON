"""Phase 0.5.9: 2D/3D atmospheres + geometry + 2D Madhu PT.

Ports of POSEIDON's geometry.py + atmosphere.py 2D/3D paths:
  - atmosphere_regions  (`geometry.py:12-87`) — 1/2/3D dimension count
  - angular_grids       (`geometry.py:90-253`) — sector / zone angular grids
  - compute_T_Madhu_2D  (`atmosphere.py:106-228`)
  - compute_T_field_*   3D evaluation tested via profiles()

Plus assign_free_params geometry params for Atmosphere_dimension in {2, 3}.
"""

import numpy as np
import pytest

from jaxposeidon import _atmosphere, _geometry


@pytest.mark.parametrize("ad,td,nem,ndn", [
    (1, None, 2, 2),
    (2, "E-M", 2, 2),
    (2, "E-M", 4, 2),
    (2, "D-N", 2, 2),
    (2, "D-N", 2, 4),
    (3, None, 2, 2),
    (3, None, 4, 4),
])
def test_atmosphere_regions_matches_poseidon(ad, td, nem, ndn):
    from POSEIDON.geometry import atmosphere_regions as p_regions
    assert _geometry.atmosphere_regions(ad, td, nem, ndn) == p_regions(ad, td, nem, ndn)


@pytest.mark.parametrize("ad,td,nem,ndn,alpha,beta,sdn,sem", [
    (1, None, 2, 2, 0.0, 0.0, False, False),
    (2, "E-M", 2, 2, 30.0, 0.0, False, False),
    (2, "E-M", 4, 2, 45.0, 0.0, False, False),
    (2, "D-N", 2, 2, 0.0, 30.0, False, False),
    (2, "E-M", 2, 2, 0.0, 0.0, False, True),
    (2, "D-N", 2, 2, 0.0, 0.0, True, False),
    (3, None, 2, 2, 30.0, 30.0, False, False),
    (3, None, 4, 4, 60.0, 45.0, False, False),
    (3, None, 2, 2, 0.0, 30.0, False, True),
    (3, None, 2, 2, 30.0, 0.0, True, False),
    (3, None, 2, 2, 0.0, 0.0, True, True),
])
def test_angular_grids_matches_poseidon(ad, td, nem, ndn, alpha, beta, sdn, sem):
    from POSEIDON.geometry import angular_grids as p_angular
    ours = _geometry.angular_grids(ad, td, nem, ndn, alpha, beta, sdn, sem)
    theirs = p_angular(ad, td, nem, ndn, alpha, beta, sdn, sem)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(a, b)


def test_compute_T_Madhu_2D_matches_poseidon():
    from POSEIDON.atmosphere import compute_T_Madhu_2D as p_madhu_2D
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
    N_sectors, N_zones = 1, 4
    phi = np.array([0.0])
    theta = np.linspace(-np.pi / 2, np.pi / 2, N_zones)
    args = (
        P, 0.5, 1.0, -2.0, -4.0, 0.4, 0.9, -1.5, -3.5,
        2000.0, 10.0, N_sectors, N_zones, 30.0, 30.0, phi, theta,
    )
    np.testing.assert_array_equal(
        _atmosphere.compute_T_Madhu_2D(*args),
        p_madhu_2D(*args),
    )


# ---------------------------------------------------------------------------
# assign_free_params geometry parity for 2D/3D
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


@pytest.mark.parametrize("ad,td,sdn,sem,expected_geom", [
    (1, None, False, False, []),
    (2, "E-M", False, False, ["alpha"]),
    (2, "D-N", False, False, ["beta"]),
    (2, "E-M", False, True, []),
    (2, "D-N", True, False, []),
    (3, None, False, False, ["alpha", "beta"]),
    (3, None, False, True, ["beta"]),
    (3, None, True, False, ["alpha"]),
    (3, None, True, True, []),
])
def test_assign_free_params_geometry_matches_poseidon(ad, td, sdn, sem, expected_geom):
    from jaxposeidon._parameter_setup import assign_free_params
    kw = dict(
        param_species=["H2O"], bulk_species=["H2", "He"],
        PT_profile="isotherm", X_profile="isochem",
        cloud_model="cloud-free",
        Atmosphere_dimension=ad, TwoD_type=td,
        sharp_DN_transition=sdn, sharp_EM_transition=sem,
    )
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(
        PT_profile="isotherm", X_profile="isochem",
        Atmosphere_dimension=ad, TwoD_type=td,
        sharp_DN_transition=sdn, sharp_EM_transition=sem,
    )
    np.testing.assert_array_equal(ours[5], theirs[5])  # geometry_params
    assert list(ours[5]) == expected_geom
    np.testing.assert_array_equal(ours[0], theirs[0])  # full param list
    np.testing.assert_array_equal(ours[-1], theirs[-1])  # cumulative
