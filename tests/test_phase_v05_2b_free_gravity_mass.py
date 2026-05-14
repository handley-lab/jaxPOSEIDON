"""Phase 0.5.2b: free gravity_setting / mass_setting parity tests.

POSEIDON's parameters.py:262-289 inserts ``log_g`` (when
``gravity_setting='free'``) or ``M_p`` (when ``mass_setting='free'``)
into the physical-parameter block immediately after the reference
parameter(s), and forbids both being free simultaneously
(parameters.py:274-275).
"""

import numpy as np
import pytest

from jaxposeidon._parameter_setup import assign_free_params, assert_v0_model_config


@pytest.mark.parametrize(
    "reference_parameter,expected_physical",
    [
        ("R_p_ref", ["R_p_ref", "log_g"]),
        ("P_ref", ["log_P_ref", "log_g"]),
        ("R_p_ref+P_ref", ["R_p_ref", "log_P_ref", "log_g"]),
    ],
)
def test_assign_free_params_free_gravity_block(reference_parameter, expected_physical):
    """`gravity_setting='free'` appends `log_g` after reference param(s)."""
    result = assign_free_params(
        param_species=[],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        reference_parameter=reference_parameter,
        gravity_setting="free",
    )
    _, physical_params, *_ = result
    assert [str(p) for p in physical_params] == expected_physical


@pytest.mark.parametrize(
    "reference_parameter,expected_physical",
    [
        ("R_p_ref", ["R_p_ref", "M_p"]),
        ("P_ref", ["log_P_ref", "M_p"]),
        ("R_p_ref+P_ref", ["R_p_ref", "log_P_ref", "M_p"]),
    ],
)
def test_assign_free_params_free_mass_block(reference_parameter, expected_physical):
    """`mass_setting='free'` appends `M_p` after reference param(s)."""
    result = assign_free_params(
        param_species=[],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        reference_parameter=reference_parameter,
        mass_setting="free",
    )
    _, physical_params, *_ = result
    assert [str(p) for p in physical_params] == expected_physical


def test_assert_v0_model_config_rejects_both_free():
    """POSEIDON parameters.py:274-275: both free is an error."""
    with pytest.raises(Exception, match="only one of mass or gravity"):
        assert_v0_model_config(
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            cloud_dim=1,
            gravity_setting="free",
            mass_setting="free",
        )


def test_assert_v0_model_config_rejects_unknown_settings():
    with pytest.raises(NotImplementedError, match="gravity_setting"):
        assert_v0_model_config(
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            cloud_dim=1,
            gravity_setting="bogus",
        )
    with pytest.raises(NotImplementedError, match="mass_setting"):
        assert_v0_model_config(
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            cloud_dim=1,
            mass_setting="bogus",
        )


def _poseidon_assign(param_species, bulk_species, **overrides):
    """POSEIDON assign_free_params is positional; pass defaults explicitly."""
    from POSEIDON.parameters import assign_free_params as p_assign
    defaults = dict(
        param_species=param_species,
        bulk_species=bulk_species,
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


def test_assign_free_params_matches_poseidon_free_gravity():
    ours = assign_free_params(
        param_species=["H2O"], bulk_species=["H2", "He"],
        PT_profile="Madhu", X_profile="isochem", cloud_model="cloud-free",
        gravity_setting="free",
    )
    theirs = _poseidon_assign(
        ["H2O"], ["H2", "He"], PT_profile="Madhu", gravity_setting="free",
    )
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
    np.testing.assert_array_equal(ours[-1], theirs[-1])


def test_assign_free_params_matches_poseidon_free_mass():
    ours = assign_free_params(
        param_species=["H2O"], bulk_species=["H2", "He"],
        PT_profile="Madhu", X_profile="isochem", cloud_model="cloud-free",
        mass_setting="free",
    )
    theirs = _poseidon_assign(
        ["H2O"], ["H2", "He"], PT_profile="Madhu", mass_setting="free",
    )
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[1], theirs[1])
    np.testing.assert_array_equal(ours[-1], theirs[-1])
