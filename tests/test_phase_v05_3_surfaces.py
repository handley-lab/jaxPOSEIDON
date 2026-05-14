"""Phase 0.5.3: surface parameter parsing + albedo file I/O + interpolation.

Ports POSEIDON `surfaces.py:17-99` and `parameters.py:1101-1124`:
- find_nearest_less_than
- load_surface_components (file I/O)
- interpolate_surface_components (linear interp onto wl grid)
- assign_free_params surface block (log_P_surf, albedo_surf,
  <component>_percentage / log_<component>_percentage)

Surface spectral coupling deferred to Phase 0.5.13d.
"""

import os

import numpy as np
import pytest

from jaxposeidon import _surface_setup
from jaxposeidon._parameter_setup import assign_free_params


def test_find_nearest_less_than_matches_poseidon():
    from POSEIDON.surfaces import find_nearest_less_than as p_find

    arr = np.array([1.0, 2.5, 3.7, 5.0, 7.2])
    for v in [0.5, 2.0, 3.6, 5.5, 8.0]:
        assert _surface_setup.find_nearest_less_than(v, arr.copy()) == p_find(
            v, arr.copy()
        )


def test_load_surface_components_requires_env_var(monkeypatch):
    monkeypatch.delenv("POSEIDON_input_data", raising=False)
    with pytest.raises(Exception, match="POSEIDON_input_data"):
        _surface_setup.load_surface_components(["any"])


def test_load_surface_components_reads_albedo_file(tmp_path, monkeypatch):
    # Synthesize an albedo file (POSEIDON expects 2-column whitespace-separated).
    refl_dir = tmp_path / "surface_reflectivities"
    refl_dir.mkdir()
    wl_lab = np.linspace(0.5, 10.0, 100)
    albedo = 0.3 + 0.1 * np.sin(wl_lab)
    np.savetxt(refl_dir / "synthetic.txt", np.column_stack([wl_lab, albedo]))
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path) + os.sep)
    data = _surface_setup.load_surface_components(["synthetic"])
    assert len(data) == 1
    np.testing.assert_array_equal(data[0][0], wl_lab)
    np.testing.assert_array_equal(data[0][1], albedo)


def test_interpolate_surface_components_matches_poseidon():
    from POSEIDON.surfaces import (
        interpolate_surface_components as p_interp,
    )

    wl_lab = np.linspace(0.5, 10.0, 100)
    albedo = 0.3 + 0.1 * np.sin(wl_lab)
    surface_component_albedos = [np.array([wl_lab, albedo])]
    wl = np.linspace(1.0, 5.0, 50)
    ours = _surface_setup.interpolate_surface_components(
        wl, ["synthetic"], surface_component_albedos
    )
    theirs = p_interp(wl, ["synthetic"], surface_component_albedos)
    for a, b in zip(ours, theirs, strict=True):
        np.testing.assert_array_equal(a, b)


def test_interpolate_surface_components_rejects_out_of_range():
    wl_lab = np.linspace(1.0, 5.0, 50)
    albedo = np.ones(50) * 0.3
    surface_component_albedos = [np.array([wl_lab, albedo])]
    wl = np.linspace(0.5, 6.0, 20)
    with pytest.raises(Exception, match="exceeds the wavelengths"):
        _surface_setup.interpolate_surface_components(
            wl, ["synthetic"], surface_component_albedos
        )


# ---------------------------------------------------------------------------
# Parameter parsing — `surface_params` block
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
    "surface,surface_model,surface_components,surface_percentage_option,expected",
    [
        (False, "gray", [], "linear", []),
        (True, "gray", [], "linear", ["log_P_surf"]),
        (True, "constant", [], "linear", ["log_P_surf", "albedo_surf"]),
        (False, "constant", [], "linear", ["albedo_surf"]),
        (
            True,
            "lab_data",
            ["basalt", "ice"],
            "linear",
            ["log_P_surf", "basalt_percentage", "ice_percentage"],
        ),
        (
            False,
            "lab_data",
            ["basalt", "ice"],
            "log",
            ["log_basalt_percentage", "log_ice_percentage"],
        ),
        (False, "lab_data", ["solo"], "linear", []),  # single component → no percentage
    ],
)
def test_assign_free_params_surface_block_matches_poseidon(
    surface,
    surface_model,
    surface_components,
    surface_percentage_option,
    expected,
):
    kw = dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        surface=surface,
        surface_model=surface_model,
        surface_components=surface_components,
        surface_percentage_option=surface_percentage_option,
    )
    ours = assign_free_params(**kw)
    theirs = _poseidon_assign(
        surface=surface,
        surface_model=surface_model,
        surface_components=surface_components,
        surface_percentage_option=surface_percentage_option,
    )
    # surface_params is element 8 in the 10-tuple
    np.testing.assert_array_equal(ours[8], theirs[8])
    assert list(ours[8]) == expected
    # Full param list parity
    np.testing.assert_array_equal(ours[0], theirs[0])
    np.testing.assert_array_equal(ours[-1], theirs[-1])
