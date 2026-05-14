"""Phase 0.5.16: high-resolution spectroscopy.

Ports POSEIDON `high_res.py:14-29, 179-256` numeric utilities:
- airtovac / vactoair (wavelength conversion)
- sysrem iterative detrending

Plus assign_free_params high_res_params block (parameters.py:1068-1090).

The wider pipeline (per-order data prep, CCF, full likelihood
prescriptions) lives in POSEIDON's high_res.py and depends on real
high-res transmission data; that integration is the follow-up.
"""

import numpy as np
import pytest

from jaxposeidon import _high_res
from jaxposeidon._parameter_setup import assign_free_params


def test_airtovac_matches_poseidon():
    from POSEIDON.high_res import airtovac as p_atv

    wl = np.linspace(0.4, 5.0, 100)
    np.testing.assert_allclose(
        _high_res.airtovac(wl),
        p_atv(wl),
        atol=0,
        rtol=1e-13,
    )


def test_vactoair_matches_poseidon():
    from POSEIDON.high_res import vactoair as p_vta

    wl = np.linspace(0.4, 5.0, 100)
    np.testing.assert_allclose(
        _high_res.vactoair(wl),
        p_vta(wl),
        atol=0,
        rtol=1e-13,
    )


def test_sysrem_matches_poseidon():
    """SYSREM iterative detrending parity (Tamuz, Mazeh & Zucker 2005)."""
    from POSEIDON.high_res import sysrem as p_sysrem
    rng = np.random.default_rng(0)
    nphi, npix = 8, 50
    # Synthetic blaze-corrected order with a few systematic components
    base = rng.standard_normal(npix) * 0.05
    phases = rng.standard_normal(nphi)
    data_array = np.outer(phases, base) + rng.standard_normal((nphi, npix)) * 0.01
    uncertainties = 0.01 * np.ones((nphi, npix))
    res_ours, U_ours = _high_res.sysrem(data_array.copy(), uncertainties.copy(), niter=3)
    res_theirs, U_theirs = p_sysrem(data_array.copy(), uncertainties.copy(), niter=3)
    np.testing.assert_allclose(res_ours, res_theirs, atol=0, rtol=1e-13)
    np.testing.assert_allclose(U_ours, U_theirs, atol=0, rtol=1e-13)


def test_airtovac_inverse_vactoair():
    wl = np.linspace(0.4, 5.0, 100)
    np.testing.assert_allclose(
        _high_res.vactoair(_high_res.airtovac(wl)),
        wl,
        atol=0,
        rtol=1e-6,
    )


# ---------------------------------------------------------------------------
# assign_free_params high_res_params block parity
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
    "method,fix_alpha,fix_W,fix_beta,fix_Delta_phi,alpha_opt,expected",
    [
        (None, False, False, True, True, "log", []),
        (
            ["CC"],
            False,
            False,
            True,
            True,
            "log",
            ["K_p", "V_sys", "W_conv", "log_alpha_HR"],
        ),
        (["CC"], False, True, True, True, "log", ["K_p", "V_sys", "log_alpha_HR"]),
        (
            ["CC"],
            False,
            False,
            False,
            False,
            "log",
            ["K_p", "V_sys", "W_conv", "Delta_phi", "log_alpha_HR", "beta_HR"],
        ),
        (
            ["CC"],
            False,
            False,
            True,
            True,
            "linear",
            ["K_p", "V_sys", "W_conv", "alpha_HR"],
        ),
        (["CC"], True, True, True, True, "log", ["K_p", "V_sys"]),
    ],
)
def test_assign_free_params_high_res_block_matches_poseidon(
    method,
    fix_alpha,
    fix_W,
    fix_beta,
    fix_Delta_phi,
    alpha_opt,
    expected,
):
    kw = dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        high_res_method=method,
        fix_alpha_high_res=fix_alpha,
        fix_W_conv_high_res=fix_W,
        fix_beta_high_res=fix_beta,
        fix_Delta_phi_high_res=fix_Delta_phi,
        alpha_high_res_option=alpha_opt,
    )
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(
        high_res_method=method,
        fix_alpha_high_res=fix_alpha,
        fix_W_conv_high_res=fix_W,
        fix_beta_high_res=fix_beta,
        fix_Delta_phi_high_res=fix_Delta_phi,
        alpha_high_res_option=alpha_opt,
    )
    # high_res_params is element 7
    np.testing.assert_array_equal(ours[7], theirs[7])
    assert list(ours[7]) == expected
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[-1], theirs[-1])
