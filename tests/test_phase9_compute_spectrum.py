"""Phase 9b: end-to-end compute_spectrum parity with POSEIDON.

Uses POSEIDON's own canonical Rayleigh oracle setup
(`POSEIDON/tests/test_TRIDENT.py`) to construct planet/star/model/
atmosphere/opac/wl, then runs both POSEIDON's `compute_spectrum` and
jaxposeidon's port — assert bit-exact equality.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon._compute_spectrum import (
    compute_spectrum as j_compute_spectrum,
    check_atmosphere_physical,
)


@pytest.fixture(scope="module", autouse=True)
def _synthetic_poseidon_input_data():
    """Provide a synthetic POSEIDON_input_data/opacity/Opacity_database_cia.hdf5
    so POSEIDON's read_opacities(..., testing=True) can open the CIA file.

    POSEIDON unconditionally opens the CIA HDF5 (absorption.py:811) even
    in testing mode. We populate just the H2-H2 group used by this test.
    """
    if os.environ.get("POSEIDON_input_data"):
        yield
        return
    with tempfile.TemporaryDirectory() as tmp:
        opac_dir = os.path.join(tmp, "opacity")
        os.makedirs(opac_dir)
        path = os.path.join(opac_dir, "Opacity_database_cia.hdf5")
        T_grid = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600,
                            1800, 2000], dtype=np.float64)
        nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
        log_cia = np.full((len(T_grid), len(nu)), -50.0, dtype=np.float64)
        with h5py.File(path, "w") as f:
            for pair in ("H2-H2", "H2-He"):
                g = f.create_group(pair)
                g.create_dataset("T", data=T_grid)
                g.create_dataset("nu", data=nu)
                g.create_dataset("log(cia)", data=log_cia)
        os.environ["POSEIDON_input_data"] = tmp
        yield
        del os.environ["POSEIDON_input_data"]


def _build_canonical_rayleigh_oracle(disable_cia=True):
    """Reproduce POSEIDON/tests/test_TRIDENT.py setup."""
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R)

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("Example Planet", R_J, mass=M_J, T_eq=1000.0)
    model = define_model("Only_Rayleigh", ["H2"], [], PT_profile="isotherm")

    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    atmosphere = make_atmosphere(
        planet, model, P, 100.0, R_J,
        np.array([1000.0]), np.array([]),
        constant_gravity=True,
    )

    wl = wl_grid_constant_R(0.2, 10.0, 10000)
    T_fine = np.arange(900, 1110, 10)
    log_P_fine = np.arange(-6.0, 2.2, 0.2)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    if disable_cia:
        opac["CIA_stored"] *= 0.0

    return planet, star, model, atmosphere, opac, wl


def test_compute_spectrum_canonical_rayleigh_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    ours = j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                              spectrum_type="transmission")
    theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                                spectrum_type="transmission")
    np.testing.assert_array_equal(ours, theirs)


def test_compute_spectrum_returns_NaN_for_unphysical_atmosphere():
    """Mirrors POSEIDON core.py:1370-1374 NaN sentinel."""
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    atmosphere["is_physical"] = False
    out = j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                             spectrum_type="transmission")
    assert out.shape == wl.shape
    assert np.all(np.isnan(out))


def test_compute_spectrum_returns_NaN_for_temperature_out_of_fine_grid():
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    # Bump T above the fine T-grid max (1100 K)
    atmosphere["T"] = np.full_like(atmosphere["T"], 1500.0)
    out = j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                             spectrum_type="transmission")
    assert np.all(np.isnan(out))


@pytest.mark.parametrize("spectrum_type", ["emission", "direct_emission",
                                            "dayside_emission",
                                            "nightside_emission",
                                            "transmission_time_average"])
def test_compute_spectrum_rejects_non_transmission_types(spectrum_type):
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    with pytest.raises(NotImplementedError, match="spectrum_type"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            spectrum_type=spectrum_type)


def test_compute_spectrum_rejects_gpu_device():
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    with pytest.raises(NotImplementedError, match="device"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            device="gpu")


def test_compute_spectrum_rejects_disable_continuum():
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    with pytest.raises(NotImplementedError, match="disable_continuum"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            disable_continuum=True)


@pytest.mark.parametrize("cloud_model", ["Iceberg", "Mie", "eddysed"])
def test_compute_spectrum_rejects_unsupported_cloud_models(cloud_model):
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    model["cloud_model"] = cloud_model
    with pytest.raises(NotImplementedError, match="cloud_model"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl)


def test_compute_spectrum_rejects_unsupported_cloud_type():
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    model["cloud_model"] = "MacMad17"
    model["cloud_type"] = "opaque_deck_plus_slab"
    with pytest.raises(NotImplementedError, match="cloud_type"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl)


def test_compute_spectrum_rejects_unsupported_cloud_dim():
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    model["cloud_model"] = "MacMad17"
    model["cloud_type"] = "deck"
    model["cloud_dim"] = 3
    with pytest.raises(NotImplementedError, match="cloud_dim"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl)


def test_compute_spectrum_unsupported_config_takes_precedence_over_NaN():
    """An unsupported spectrum_type must raise even when atmosphere is
    also unphysical — descriptive NotImplementedError beats NaN."""
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    atmosphere["T"] = np.full_like(atmosphere["T"], 5000.0)
    with pytest.raises(NotImplementedError, match="spectrum_type"):
        j_compute_spectrum(planet, star, model, atmosphere, opac, wl,
                            spectrum_type="emission")


def test_check_atmosphere_physical_handles_numpy_bool():
    """np.bool_(False) in 'is_physical' triggers the unphysical branch."""
    _, _, _, atmosphere, opac, _ = _build_canonical_rayleigh_oracle()
    atmosphere["is_physical"] = np.bool_(False)
    assert check_atmosphere_physical(atmosphere, opac) is False


def test_compute_spectrum_with_active_species_and_CIA_matches_poseidon():
    """End-to-end POSEIDON parity with non-empty active_species + CIA on."""
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (create_star, create_planet, define_model,
                                make_atmosphere, read_opacities,
                                wl_grid_constant_R,
                                compute_spectrum as p_compute_spectrum)
    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=M_J, T_eq=1000.0)
    # Active H2O species, isochem free chemistry
    model = define_model("active_test", ["H2", "He"], ["H2O"],
                          PT_profile="isotherm", X_profile="isochem")
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 50)
    atmosphere = make_atmosphere(planet, model, P, 10.0, R_J,
                                  np.array([1000.0]),
                                  np.array([-4.0]),  # log_X(H2O) = -4
                                  constant_gravity=True)
    wl = wl_grid_constant_R(1.0, 3.0, 500)
    T_fine = np.arange(800, 1210, 20)
    log_P_fine = np.arange(-6.0, 2.2, 0.4)
    opac = read_opacities(model, wl, "opacity_sampling", T_fine, log_P_fine,
                          testing=True)
    # CIA stays on (synthetic CIA HDF5 from fixture); sigma_stored is zero
    # in testing mode but the active-species code path is fully exercised.
    ours = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    theirs = p_compute_spectrum(planet, star, model, atmosphere, opac, wl)
    np.testing.assert_allclose(ours, theirs, atol=1e-15, rtol=1e-13)


def test_check_atmosphere_physical_matches_poseidon():
    from POSEIDON.core import check_atmosphere_physical as p_check
    planet, star, model, atmosphere, opac, wl = (
        _build_canonical_rayleigh_oracle()
    )
    assert check_atmosphere_physical(atmosphere, opac) == p_check(atmosphere, opac)

    atmosphere["is_physical"] = False
    assert check_atmosphere_physical(atmosphere, opac) == p_check(atmosphere, opac)

    atmosphere["is_physical"] = True
    atmosphere["T"] = np.full_like(atmosphere["T"], 5000.0)
    assert check_atmosphere_physical(atmosphere, opac) == p_check(atmosphere, opac)
