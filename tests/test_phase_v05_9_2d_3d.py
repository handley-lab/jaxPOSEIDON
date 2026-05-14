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


@pytest.mark.parametrize(
    "ad,td,nem,ndn",
    [
        (1, None, 2, 2),
        (2, "E-M", 2, 2),
        (2, "E-M", 4, 2),
        (2, "D-N", 2, 2),
        (2, "D-N", 2, 4),
        (3, None, 2, 2),
        (3, None, 4, 4),
    ],
)
def test_atmosphere_regions_matches_poseidon(ad, td, nem, ndn):
    from POSEIDON.geometry import atmosphere_regions as p_regions

    assert _geometry.atmosphere_regions(ad, td, nem, ndn) == p_regions(ad, td, nem, ndn)


@pytest.mark.parametrize(
    "ad,td,nem,ndn,alpha,beta,sdn,sem",
    [
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
    ],
)
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
        P,
        0.5,
        1.0,
        -2.0,
        -4.0,
        0.4,
        0.9,
        -1.5,
        -3.5,
        2000.0,
        10.0,
        N_sectors,
        N_zones,
        30.0,
        30.0,
        phi,
        theta,
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
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        object_type="transiting",
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_type="deck",
        gravity_setting="fixed",
        mass_setting="fixed",
        stellar_contam=None,
        offsets_applied=None,
        error_inflation=None,
        PT_dim=1,
        X_dim=1,
        cloud_dim=1,
        TwoD_type=None,
        TwoD_param_scheme="difference",
        species_EM_gradient=[],
        species_DN_gradient=[],
        species_vert_gradient=[],
        Atmosphere_dimension=1,
        opaque_Iceberg=False,
        surface=False,
        sharp_DN_transition=False,
        sharp_EM_transition=False,
        reference_parameter="R_p_ref",
        disable_atmosphere=False,
        aerosol_species=[],
        log_P_slope_arr=[-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0],
        number_P_knots=0,
        PT_penalty=False,
        high_res_method=None,
        alpha_high_res_option="log",
        fix_alpha_high_res=False,
        fix_W_conv_high_res=False,
        fix_beta_high_res=True,
        fix_Delta_phi_high_res=True,
        lognormal_logwidth_free=False,
        surface_components=[],
        surface_model="gray",
        surface_percentage_option="linear",
        thermal=True,
        reflection=False,
    )
    defaults.update(overrides)
    return p_assign(**defaults)


# ---------------------------------------------------------------------------
# profiles() 2D/3D end-to-end parity
# ---------------------------------------------------------------------------
def _common_atm_args_multi(N_layers=50, N_sectors=1, N_zones=1, alpha=0.0, beta=0.0):
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    if N_sectors > 1 and N_zones > 1:
        ad, td = 3, None
        nem, ndn = N_sectors - 2, N_zones - 2
    elif N_sectors > 1:
        ad, td = 2, "E-M"
        nem, ndn = N_sectors - 2, 2
    elif N_zones > 1:
        ad, td = 2, "D-N"
        nem, ndn = 2, N_zones - 2
    else:
        ad, td = 1, None
        nem, ndn = 2, 2
    phi, theta, *_ = _geometry.angular_grids(
        ad,
        td,
        nem,
        ndn,
        alpha,
        beta,
        False,
        False,
    )
    return dict(
        P=P,
        R_p=7.1492e7,
        g_0=24.79,
        P_ref=10.0,
        R_p_ref=7.1492e7,
        N_sectors=N_sectors,
        N_zones=N_zones,
        alpha=alpha,
        beta=beta,
        phi=phi,
        theta=theta,
        species_vert_gradient=np.array([], dtype=str),
        He_fraction=0.17,
        P_param_set=1.0e-6,
        log_P_slope_phot=0.5,
        log_P_slope_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
        constant_gravity=False,
    )


def _profiles_assert_match_multi(cfg):
    from POSEIDON.atmosphere import profiles as p_profiles

    ours = _atmosphere.profiles(**cfg)
    theirs = p_profiles(
        cfg["P"],
        cfg["R_p"],
        cfg["g_0"],
        cfg["PT_profile"],
        cfg["X_profile"],
        cfg["PT_state"],
        cfg["P_ref"],
        cfg["R_p_ref"],
        cfg["log_X_state"],
        cfg["included_species"],
        cfg["bulk_species"],
        cfg["param_species"],
        cfg["active_species"],
        cfg["CIA_pairs"],
        cfg["ff_pairs"],
        cfg["bf_species"],
        cfg["N_sectors"],
        cfg["N_zones"],
        cfg["alpha"],
        cfg["beta"],
        cfg["phi"],
        cfg["theta"],
        cfg["species_vert_gradient"],
        cfg["He_fraction"],
        cfg.get("T_input", None),
        cfg.get("X_input", None),
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
            assert a == b
        else:
            np.testing.assert_allclose(
                a,
                b,
                atol=0,
                rtol=1e-13,
                err_msg=f"profiles() output {i} differs",
            )


def test_profiles_isotherm_with_Nsectors_gt_1_is_undefined():
    """Calling profiles() with a 1D PT profile and N_sectors > 1 (or N_zones > 1)
    is not a meaningful POSEIDON configuration — the convention is
    `Atmosphere_dimension == max(PT_dim, X_dim)`, so multi-dim geometry
    requires PT_dim or X_dim > 1. POSEIDON's numba radial_profiles silently
    reads out-of-bounds memory in this case and produces nonsense (inf/zero)
    in n; the jaxposeidon port broadcasts T/mu/X to (N_sectors, N_zones) and
    instead returns finite but physically meaningless replicated values.
    Neither behaviour is a parity bug; we therefore do not parity-test this
    undefined configuration. Real 2D/3D parity is exercised below via the
    2D Madhu test where PT_state has shape consistent with N_zones > 1.
    """
    cfg = _common_atm_args_multi(N_sectors=4, N_zones=1, alpha=30.0, beta=0.0)
    cfg.update(
        PT_profile="isotherm",
        X_profile="isochem",
        PT_state=np.array([1200.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    out = _atmosphere.profiles(**cfg)
    # Sanity: the port still produces a finite atmosphere tuple (no crash).
    assert out[-1] is True
    assert out[0].shape == (50, 1, 1)  # T kept 1D as POSEIDON does
    assert out[1].shape == (50, 4, 1)  # n broadcast to (N_sectors, N_zones)


def test_profiles_Madhu_2D_DN_matches_poseidon():
    cfg = _common_atm_args_multi(N_sectors=1, N_zones=4, alpha=0.0, beta=30.0)
    cfg.update(
        PT_profile="Madhu",
        X_profile="isochem",
        PT_state=np.array([0.5, 1.0, -2.0, -4.0, 0.4, 0.9, -1.5, -3.5, 2000.0]),
        log_X_state=np.array([[-3.0, 0.0, 0.0, -3.0]]),
        included_species=np.array(["H2", "He", "H2O"]),
        bulk_species=np.array(["H2", "He"]),
        param_species=np.array(["H2O"]),
        active_species=np.array(["H2O"]),
        CIA_pairs=np.array([], dtype=str),
        ff_pairs=np.array([], dtype=str),
        bf_species=np.array([], dtype=str),
    )
    _profiles_assert_match_multi(cfg)


def test_profiles_lever_rejects_multi_dim():
    cfg = _common_atm_args_multi(N_sectors=4, N_zones=1, alpha=30.0, beta=0.0)
    cfg.update(
        PT_profile="isotherm",
        X_profile="lever",
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
    with pytest.raises(Exception, match="Lever"):
        _atmosphere.profiles(**cfg)


@pytest.mark.parametrize(
    "ad,td,sdn,sem,expected_geom",
    [
        (1, None, False, False, []),
        (2, "E-M", False, False, ["alpha"]),
        (2, "D-N", False, False, ["beta"]),
        (2, "E-M", False, True, []),
        (2, "D-N", True, False, []),
        (3, None, False, False, ["alpha", "beta"]),
        (3, None, False, True, ["beta"]),
        (3, None, True, False, ["alpha"]),
        (3, None, True, True, []),
    ],
)
def test_assign_free_params_geometry_matches_poseidon(ad, td, sdn, sem, expected_geom):
    from jaxposeidon._parameter_setup import assign_free_params

    kw = dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        Atmosphere_dimension=ad,
        TwoD_type=td,
        sharp_DN_transition=sdn,
        sharp_EM_transition=sem,
    )
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(
        PT_profile="isotherm",
        X_profile="isochem",
        Atmosphere_dimension=ad,
        TwoD_type=td,
        sharp_DN_transition=sdn,
        sharp_EM_transition=sem,
    )
    np.testing.assert_array_equal(ours[5], theirs[5])  # geometry_params
    assert list(ours[5]) == expected_geom
    np.testing.assert_array_equal(ours[0], theirs[0])  # full param list
    np.testing.assert_array_equal(ours[-1], theirs[-1])  # cumulative
