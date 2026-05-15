"""Phase 0.5.13d: surface spectral coupling in single-stream emission parity.

Ports POSEIDON `core.py:1741-1830` (`surf_reflect` construction),
`emission.py:1681-1878` (`assign_assumptions_and_compute_single_stream_emission`),
and `absorption.py:1192-1196` (`enable_surface` kappa contribution).

Test cases:
- surface=True, surface_model='gray' → emission_single_stream_w_albedo.
- surface=True, surface_model='constant' (free albedo_surf).
- surface=True, surface_model='lab_data' (single component, percentages on
  albedos vs models).
- disable_atmosphere=True (bare-rock).
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute_spectrum


@pytest.fixture(scope="module", autouse=True)
def _synthetic_poseidon_input_data():
    """Provide a synthetic POSEIDON_input_data + CIA HDF5 + a surface_reflectivities
    folder with a synthetic lab-data albedo so `surface_model='lab_data'` works."""
    if os.environ.get("POSEIDON_input_data"):
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        opac_dir = os.path.join(tmp, "opacity")
        os.makedirs(opac_dir)
        path = os.path.join(opac_dir, "Opacity_database_cia.hdf5")
        T_grid = np.array(
            [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000], dtype=np.float64
        )
        nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
        log_cia = np.full((len(T_grid), len(nu)), -50.0, dtype=np.float64)
        with h5py.File(path, "w") as f:
            for pair in ("H2-H2", "H2-He"):
                g = f.create_group(pair)
                g.create_dataset("T", data=T_grid)
                g.create_dataset("nu", data=nu)
                g.create_dataset("log(cia)", data=log_cia)

        refl_dir = os.path.join(tmp, "surface_reflectivities")
        os.makedirs(refl_dir)
        wl_lab = np.linspace(0.1, 30.0, 500)
        for name, base in (("synthetic_a", 0.25), ("synthetic_b", 0.55)):
            albedo = base + 0.1 * np.sin(wl_lab)
            np.savetxt(
                os.path.join(refl_dir, f"{name}.txt"), np.column_stack([wl_lab, albedo])
            )

        os.environ["POSEIDON_input_data"] = tmp + os.sep
        yield
        del os.environ["POSEIDON_input_data"]


def _build_oracle(
    surface=False,
    surface_model="gray",
    surface_components=(),
    surface_percentage_apply_to="models",
    disable_atmosphere=False,
):
    """POSEIDON setup with optional surface / disable_atmosphere."""
    from POSEIDON.constants import M_J, R_J, R_Sun
    from POSEIDON.core import (
        create_planet,
        create_star,
        define_model,
        make_atmosphere,
        read_opacities,
        wl_grid_constant_R,
    )

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("Example Planet", R_J, mass=M_J, T_eq=1000.0)
    model = define_model(
        "Surface_Rayleigh",
        ["H2"],
        [],
        PT_profile="isotherm",
        surface=surface,
        surface_model=surface_model,
        surface_components=list(surface_components),
        surface_percentage_apply_to=surface_percentage_apply_to,
        disable_atmosphere=disable_atmosphere,
    )

    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)

    surface_params = []
    if disable_atmosphere:
        surface_params.append(500.0)  # T_surf (bare-rock)
    elif surface:
        surface_params.append(-2.0)  # log_P_surf
    if surface_model == "constant":
        surface_params.append(0.4)  # albedo_surf
    if surface_model == "lab_data" and len(surface_components) > 1:
        # Equal partitioning across components
        for _ in surface_components:
            surface_params.append(1.0 / len(surface_components))

    atmosphere = make_atmosphere(
        planet,
        model,
        P,
        100.0,
        R_J,
        np.array([1000.0]),
        np.array([]),
        surface_params=np.array(surface_params),
        constant_gravity=True,
    )

    wl = wl_grid_constant_R(0.5, 20.0, 2000)
    T_fine = np.arange(900, 1110, 10)
    log_P_fine = np.arange(-6.0, 2.2, 0.2)
    opac = read_opacities(
        model, wl, "opacity_sampling", T_fine, log_P_fine, testing=True
    )
    opac["CIA_stored"] *= 0.0

    star["F_star"] = np.full_like(wl, 1e7)
    star["wl_star"] = wl

    return planet, star, model, atmosphere, opac, wl


def _assert_parity(ours, theirs):
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_surface_gray_emission_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_oracle(
        surface=True, surface_model="gray"
    )
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    _assert_parity(ours, theirs)


def test_surface_constant_emission_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_oracle(
        surface=True, surface_model="constant"
    )
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    _assert_parity(ours, theirs)


def test_surface_lab_data_apply_to_albedos_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_oracle(
        surface=True,
        surface_model="lab_data",
        surface_components=("synthetic_a", "synthetic_b"),
        surface_percentage_apply_to="albedos",
    )
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    _assert_parity(ours, theirs)


def test_surface_lab_data_apply_to_models_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_oracle(
        surface=True,
        surface_model="lab_data",
        surface_components=("synthetic_a", "synthetic_b"),
        surface_percentage_apply_to="models",
    )
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="emission"
    )
    _assert_parity(ours, theirs)


def test_disable_atmosphere_bare_rock_direct_emission_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_oracle(
        surface=True,
        surface_model="constant",
        disable_atmosphere=True,
    )
    planet["system_distance"] = 10.0 * 3.086e16  # 10 pc
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="direct_emission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="direct_emission"
    )
    _assert_parity(ours, theirs)
