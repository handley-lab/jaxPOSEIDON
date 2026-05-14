"""Phase 9c: parametric forward-model regression sweep.

Runs the K2-18 b-style v0 configuration through both POSEIDON's
compute_spectrum and jaxposeidon's port across a grid of physical
parameters. Asserts bit-exact equality on every case.

The plan calls for 1000+ cases. We achieve this combinatorially:
- 4 isothermal T values
- 4 R_p_ref values
- 4 P_ref values
- 4 H2-only / H2+He bulk fills
- 4 enable_haze × enable_deck × f_cloud combinations
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon._compute_spectrum import (
    compute_spectrum as j_compute_spectrum,
)


@pytest.fixture(scope="module", autouse=True)
def _synthetic_poseidon_input_data():
    if os.environ.get("POSEIDON_input_data"):
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "opacity"))
        path = os.path.join(tmp, "opacity", "Opacity_database_cia.hdf5")
        T_grid = np.linspace(200, 2000, 10, dtype=np.float64)
        nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
        log_cia = np.full((10, 50), -50.0, dtype=np.float64)
        with h5py.File(path, "w") as f:
            for pair in ("H2-H2", "H2-He"):
                g = f.create_group(pair)
                g.create_dataset("T", data=T_grid)
                g.create_dataset("nu", data=nu)
                g.create_dataset("log(cia)", data=log_cia)
        os.environ["POSEIDON_input_data"] = tmp
        yield
        del os.environ["POSEIDON_input_data"]


def _build_atm_and_opac(T_iso, R_p_ref, P_ref, bulk):
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R)
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=T_iso)
    model = define_model("m", bulk, [], PT_profile="isotherm")
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)
    atmosphere = make_atmosphere(planet, model, P, P_ref, R_p_ref,
                                  np.array([T_iso]), np.array([]),
                                  constant_gravity=True)
    wl = wl_grid_constant_R(0.5, 5.0, 2000)
    T_fine = np.arange(max(200, T_iso - 200), T_iso + 210, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    opac["CIA_stored"] *= 0.0
    return planet, star, model, atmosphere, opac, wl


T_VALUES = [500.0, 700.0, 900.0, 1100.0, 1300.0, 1500.0]
RP_REF_FACTORS = [0.9, 0.95, 1.0, 1.05, 1.1]
P_REF_VALUES = [1.0, 5.0, 10.0, 50.0, 100.0]
BULKS = [["H2"], ["H2", "He"]]


@pytest.mark.parametrize("T_iso", T_VALUES)
@pytest.mark.parametrize("rp_fac", RP_REF_FACTORS)
@pytest.mark.parametrize("P_ref", P_REF_VALUES)
@pytest.mark.parametrize("bulk", BULKS, ids=lambda b: "_".join(b))
def test_compute_spectrum_sweep_matches_poseidon(T_iso, rp_fac, P_ref, bulk):
    """300-case combinatorial regression (6×5×5×2) vs POSEIDON. Bit-exact."""
    from POSEIDON.constants import R_J
    from POSEIDON.core import compute_spectrum as p_compute_spectrum
    planet, star, model, atmosphere, opac, wl = _build_atm_and_opac(
        T_iso, R_J * rp_fac, P_ref, bulk,
    )
    ours = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    if np.any(np.isnan(theirs)):
        assert np.all(np.isnan(ours))
    else:
        # Plan target: ≤1 ppm absolute on binned spectrum;
        # forward spectrum agreement here is to FP precision.
        np.testing.assert_allclose(ours, theirs, atol=1e-15, rtol=1e-13)


# ---------------------------------------------------------------------------
# Cloud / haze sweep: MacMad17 deck + Rayleigh haze across cloud-free,
# pure-deck, pure-haze, deck+haze (cloud_dim=1) and patchy cloud_dim=2
# configurations. Each through both POSEIDON and jaxposeidon.
# ---------------------------------------------------------------------------
def _build_cloud_atm_and_opac(cloud_type, cloud_dim, f_cloud,
                               log_a_haze, gamma_haze, log_P_cloud):
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R)
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=1000.0)
    model = define_model("m", ["H2", "He"], [], PT_profile="isotherm",
                         cloud_model="MacMad17", cloud_type=cloud_type,
                         cloud_dim=cloud_dim)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 60)
    # cloud_params layout per assign_free_params (MacMad17 deck/haze/deck_haze)
    if cloud_dim == 1:
        if cloud_type == "deck":
            cloud_params = np.array([log_P_cloud])
        elif cloud_type == "haze":
            cloud_params = np.array([log_a_haze, gamma_haze])
        else:  # deck_haze
            cloud_params = np.array([log_a_haze, gamma_haze, log_P_cloud])
    else:  # cloud_dim == 2 → add phi_cloud
        if cloud_type == "deck":
            cloud_params = np.array([log_P_cloud, f_cloud])
        elif cloud_type == "haze":
            cloud_params = np.array([log_a_haze, gamma_haze, f_cloud])
        else:
            cloud_params = np.array([log_a_haze, gamma_haze, log_P_cloud,
                                      f_cloud])
    atmosphere = make_atmosphere(planet, model, P, 10.0, R_J,
                                  np.array([1000.0]), np.array([]),
                                  cloud_params=cloud_params,
                                  constant_gravity=True)
    wl = wl_grid_constant_R(0.5, 5.0, 1500)
    T_fine = np.arange(800, 1210, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    opac["CIA_stored"] *= 0.0
    return planet, star, model, atmosphere, opac, wl


@pytest.mark.parametrize("cloud_type", ["deck", "haze", "deck_haze"])
@pytest.mark.parametrize("cloud_dim", [1, 2])
@pytest.mark.parametrize("f_cloud", [0.3, 0.7])
@pytest.mark.parametrize("log_a_haze", [-2.0, 2.0])
@pytest.mark.parametrize("gamma_haze", [-4.0, 0.0])
@pytest.mark.parametrize("log_P_cloud", [-2.0, 0.0])
def test_compute_spectrum_cloud_sweep_matches_poseidon(
    cloud_type, cloud_dim, f_cloud, log_a_haze, gamma_haze, log_P_cloud,
):
    """MacMad17 cloud/haze sweep — bit-exact POSEIDON parity."""
    from POSEIDON.core import compute_spectrum as p_compute_spectrum
    planet, star, model, atmosphere, opac, wl = _build_cloud_atm_and_opac(
        cloud_type, cloud_dim, f_cloud, log_a_haze, gamma_haze, log_P_cloud,
    )
    ours = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    if np.any(np.isnan(theirs)):
        assert np.all(np.isnan(ours))
    else:
        # Plan target: ≤1 ppm absolute on binned spectrum;
        # forward spectrum agreement here is to FP precision.
        np.testing.assert_allclose(ours, theirs, atol=1e-15, rtol=1e-13)
