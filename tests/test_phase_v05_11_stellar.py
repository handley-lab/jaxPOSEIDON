"""Phase 0.5.11: stellar contamination forward model.

Ports POSEIDON `stellar.py:26-794`:
- planck_lambda
- stellar_contamination_single_spot (Rackham+17/18)
- stellar_contamination_general

Plus assign_free_params stellar_params block per parameters.py:1016-1031.

The pysynphot / PyMSG grid loaders live in `_stellar_grid_loader.py`;
their parity is intentionally not POSEIDON-asserted here (the external
grids are env-gated and not available in CI). Synthetic spectra are
used downstream of the loader.
"""

import os

import numpy as np
import pytest

from jaxposeidon import _stellar
from jaxposeidon._parameter_setup import assign_free_params


def test_planck_lambda_matches_poseidon():
    from POSEIDON.stellar import planck_lambda as p_planck

    T = 5800.0
    wl = np.linspace(0.3, 5.0, 200)
    np.testing.assert_allclose(
        _stellar.planck_lambda(T, wl),
        p_planck(T, wl),
        atol=0,
        rtol=1e-13,
    )


def test_stellar_contamination_single_spot_matches_poseidon():
    from POSEIDON.stellar import stellar_contamination_single_spot as p_sc

    wl = np.linspace(0.5, 5.0, 100)
    I_phot = np.asarray(_stellar.planck_lambda(5800.0, wl))
    I_het = np.asarray(_stellar.planck_lambda(4500.0, wl))
    f = 0.1
    np.testing.assert_allclose(
        np.asarray(_stellar.stellar_contamination_single_spot(f, I_het, I_phot)),
        p_sc(f, I_het, I_phot),
        atol=0,
        rtol=1e-13,
    )


def test_stellar_contamination_general_matches_poseidon():
    from POSEIDON.stellar import stellar_contamination_general as p_sc

    wl = np.linspace(0.5, 5.0, 100)
    I_phot = np.asarray(_stellar.planck_lambda(5800.0, wl))
    I_het = np.stack(
        [
            np.asarray(_stellar.planck_lambda(4500.0, wl)),
            np.asarray(_stellar.planck_lambda(6200.0, wl)),
        ]
    )
    f_het = np.array([0.08, 0.03])
    np.testing.assert_allclose(
        np.asarray(_stellar.stellar_contamination_general(f_het, I_het, I_phot)),
        p_sc(f_het, I_het, I_phot),
        atol=0,
        rtol=1e-13,
    )


def test_single_spot_factor_is_unity_when_no_spot():
    wl = np.linspace(0.5, 5.0, 50)
    I_phot = _stellar.planck_lambda(5800.0, wl)
    I_het = _stellar.planck_lambda(4500.0, wl)
    eps = _stellar.stellar_contamination_single_spot(0.0, I_het, I_phot)
    np.testing.assert_allclose(eps, 1.0, atol=0, rtol=1e-15)


# ---------------------------------------------------------------------------
# assign_free_params stellar_params parity
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


@pytest.mark.parametrize(
    "stellar_contam,expected",
    [
        (None, []),
        ("one_spot", ["f_het", "T_het", "T_phot"]),
        (
            "one_spot_free_log_g",
            ["f_het", "T_het", "T_phot", "log_g_het", "log_g_phot"],
        ),
        ("two_spots", ["f_spot", "f_fac", "T_spot", "T_fac", "T_phot"]),
        (
            "two_spots_free_log_g",
            [
                "f_spot",
                "f_fac",
                "T_spot",
                "T_fac",
                "T_phot",
                "log_g_spot",
                "log_g_fac",
                "log_g_phot",
            ],
        ),
    ],
)
def test_assign_free_params_stellar_block_matches_poseidon(stellar_contam, expected):
    kw = dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        stellar_contam=stellar_contam,
    )
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(stellar_contam=stellar_contam)
    # stellar_params is element 6
    np.testing.assert_array_equal(ours[6], theirs[6])
    assert list(ours[6]) == expected
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[-1], theirs[-1])


# ---------------------------------------------------------------------------
# Env-gated parity tests against the real pysynphot / PyMSG grids.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    os.environ.get("PYSYN_CDBS") is None,
    reason="pysynphot grids unavailable; set $PYSYN_CDBS to enable.",
)
def test_load_stellar_pysynphot_matches_poseidon():
    from jaxposeidon._stellar_grid_loader import load_stellar_pysynphot
    from POSEIDON.stellar import load_stellar_pysynphot as p_load

    wl = np.linspace(0.5, 5.0, 200)
    np.testing.assert_allclose(
        load_stellar_pysynphot(wl, 5800.0, 0.0, 4.4),
        p_load(wl, 5800.0, 0.0, 4.4),
        atol=0,
        rtol=1e-13,
    )


@pytest.mark.skipif(
    os.environ.get("MSG_DIR") is None,
    reason="PyMSG grids unavailable; set $MSG_DIR to enable.",
)
def test_load_stellar_pymsg_matches_poseidon():
    from jaxposeidon._stellar_grid_loader import (
        load_stellar_pymsg,
        open_pymsg_grid,
    )
    from POSEIDON.stellar import (
        load_stellar_pymsg as p_load,
        open_pymsg_grid as p_open,
    )

    wl = np.linspace(0.5, 5.0, 200)
    grid_name = "Goettingen-HiRes"
    sg_ours = open_pymsg_grid(grid_name)
    sg_theirs = p_open(grid_name)
    np.testing.assert_allclose(
        load_stellar_pymsg(wl, sg_ours, 5800.0, 0.0, 4.4, grid_name),
        p_load(wl, sg_theirs, 5800.0, 0.0, 4.4, grid_name),
        atol=0,
        rtol=1e-13,
    )


def test_assign_free_params_rejects_unknown_stellar_contam():
    with pytest.raises(NotImplementedError, match="stellar_contam"):
        assign_free_params(
            param_species=["H2O"],
            bulk_species=["H2", "He"],
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            stellar_contam="unknown_spot",
        )
