"""Phase 0.5.14: eddysed cloud-model dispatch parity vs POSEIDON.

POSEIDON's `compute_spectrum` overwrites the runtime `kappa_cloud`
extinction array with the user-supplied `kappa_cloud_eddysed` whenever
`model['cloud_model'] == 'eddysed'`
(`POSEIDON/core.py:1685-1700`).  These tests construct the canonical
Rayleigh oracle, attach pre-computed eddysed arrays into the atmosphere
dict, then run POSEIDON's `compute_spectrum` and jaxposeidon's port and
assert bit-exact equality.

The PICASO-file path (HDF5) is exercised by a separate test that writes
a small synthetic file in `tmp_path` and checks the loader round-trips
the contents into the atmosphere dict.
"""

import os
import tempfile

import h5py
import numpy as np
import pytest

from jaxposeidon._compute_spectrum import compute_spectrum as j_compute_spectrum
from jaxposeidon._eddysed_input_loader import (
    read_eddysed_file,
    reshape_eddysed_for_atmosphere,
)


@pytest.fixture(scope="module", autouse=True)
def _synthetic_poseidon_input_data():
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
        os.environ["POSEIDON_input_data"] = tmp
        yield
        del os.environ["POSEIDON_input_data"]


def _build_eddysed_oracle():
    """Canonical Rayleigh oracle with eddysed cloud arrays injected.

    POSEIDON's `make_atmosphere` reaches `unpack_cloud_params` which for
    `cloud_model='eddysed'` returns scalar dummy values
    (`parameters.py:2444-2473`). The test then overwrites those slots in
    the atmosphere dict with correctly-shaped (N_layers, N_sectors,
    N_zones, N_wl) eddysed arrays before calling `compute_spectrum`.
    """
    from POSEIDON.constants import R_Sun, R_J, M_J
    from POSEIDON.core import (
        create_star,
        create_planet,
        define_model,
        make_atmosphere,
        read_opacities,
        wl_grid_constant_R,
    )

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("Example Planet", R_J, mass=M_J, T_eq=1000.0)
    model = define_model(
        "eddysed_test",
        ["H2"],
        [],
        PT_profile="isotherm",
        cloud_model="eddysed",
        cloud_type="deck",
    )

    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 100)
    cloud_params = np.array([0.0, 0.0, 0.0])
    atmosphere = make_atmosphere(
        planet,
        model,
        P,
        100.0,
        R_J,
        np.array([1000.0]),
        np.array([]),
        cloud_params=cloud_params,
        constant_gravity=True,
    )

    wl = wl_grid_constant_R(0.2, 10.0, 10000)
    T_fine = np.arange(900, 1110, 10)
    log_P_fine = np.arange(-6.0, 2.2, 0.2)
    opac = read_opacities(
        model, wl, "opacity_sampling", T_fine, log_P_fine, testing=True
    )
    opac["CIA_stored"] *= 0.0

    N_layers = len(P)
    N_wl = len(wl)
    rng = np.random.default_rng(0)
    kappa = rng.uniform(1.0e-8, 1.0e-6, size=(N_layers, 1, 1, N_wl))
    g = np.full((N_layers, 1, 1, N_wl), 0.7)
    w = np.full((N_layers, 1, 1, N_wl), 0.5)
    atmosphere["kappa_cloud_eddysed"] = kappa
    atmosphere["g_cloud_eddysed"] = g
    atmosphere["w_cloud_eddysed"] = w

    return planet, star, model, atmosphere, opac, wl


def test_compute_spectrum_eddysed_matches_poseidon():
    from POSEIDON.core import compute_spectrum as p_compute_spectrum

    planet, star, model, atmosphere, opac, wl = _build_eddysed_oracle()
    ours = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
    )
    theirs = p_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
    )
    np.testing.assert_array_equal(ours, theirs)


def test_compute_spectrum_eddysed_overrides_runtime_kappa_cloud():
    """The eddysed array must REPLACE the runtime extinction kappa_cloud:
    a different eddysed kappa must give a different spectrum, while the
    other extinction sources are unchanged."""
    planet, star, model, atmosphere, opac, wl = _build_eddysed_oracle()
    s_a = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)

    atmosphere["kappa_cloud_eddysed"] = atmosphere["kappa_cloud_eddysed"] * 10.0
    s_b = j_compute_spectrum(planet, star, model, atmosphere, opac, wl)

    assert not np.array_equal(s_a, s_b)


def test_eddysed_input_loader_roundtrip(tmp_path):
    """Loader returns the same arrays that were written, and the reshape
    helper broadcasts a (N_layers, N_wl) PICASO-style payload to the
    (N_layers, N_sectors, N_zones, N_wl) layout POSEIDON consumes."""
    N_layers, N_wl = 8, 16
    rng = np.random.default_rng(42)
    kappa = rng.uniform(1.0e-8, 1.0e-6, size=(N_layers, N_wl))
    g = rng.uniform(0.0, 1.0, size=(N_layers, N_wl))
    w = rng.uniform(0.0, 1.0, size=(N_layers, N_wl))
    P = np.logspace(-6, 2, N_layers)
    wavelength = np.linspace(1.0, 10.0, N_wl)

    path = tmp_path / "eddysed_picaso.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("kappa_cloud", data=kappa)
        f.create_dataset("g_cloud", data=g)
        f.create_dataset("w_cloud", data=w)
        f.create_dataset("P", data=P)
        f.create_dataset("wavelength", data=wavelength)

    payload = read_eddysed_file(str(path))
    np.testing.assert_array_equal(payload["kappa_cloud"], kappa)
    np.testing.assert_array_equal(payload["g_cloud"], g)
    np.testing.assert_array_equal(payload["w_cloud"], w)
    np.testing.assert_array_equal(payload["P"], P)
    np.testing.assert_array_equal(payload["wavelength"], wavelength)

    k4, g4, w4 = reshape_eddysed_for_atmosphere(
        payload["kappa_cloud"], payload["g_cloud"], payload["w_cloud"]
    )
    assert k4.shape == (N_layers, 1, 1, N_wl)
    assert g4.shape == (N_layers, 1, 1, N_wl)
    assert w4.shape == (N_layers, 1, 1, N_wl)
    np.testing.assert_array_equal(k4[:, 0, 0, :], kappa)
    np.testing.assert_array_equal(g4[:, 0, 0, :], g)
    np.testing.assert_array_equal(w4[:, 0, 0, :], w)


def _poseidon_defaults(**overrides):
    """POSEIDON.parameters.assign_free_params requires all 36 positional
    args; this dict mirrors the same defaults used in
    test_phase_v05_11_stellar.py."""
    defaults = dict(
        param_species=[],
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
    return defaults


@pytest.mark.parametrize("cloud_dim", [1, 2])
def test_assign_free_params_eddysed_parity(cloud_dim):
    """Mirror POSEIDON parameters.py:978-985 eddysed parameter ordering."""
    from POSEIDON.parameters import assign_free_params as p_assign

    from jaxposeidon._parameter_setup import assign_free_params as j_assign

    kw = _poseidon_defaults(cloud_model="eddysed", cloud_dim=cloud_dim)
    j_params, *_ = j_assign(**kw)
    p_params, *_ = p_assign(**kw)
    np.testing.assert_array_equal(j_params, p_params)


def test_compute_spectrum_eddysed_skipped_when_kappa_contributions_provided():
    """Mirrors POSEIDON: the eddysed overwrite lives inside the
    extinction() branch only (core.py:1685-1700), so when the caller
    supplies pre-built kappa_contributions the eddysed dispatch is
    bypassed."""
    planet, star, model, atmosphere, opac, wl = _build_eddysed_oracle()
    N_layers, _, _, N_wl = atmosphere["kappa_cloud_eddysed"].shape
    contributions = (
        np.ones((N_layers, 1, 1, N_wl)) * 1.0e-12,
        np.ones((N_layers, 1, 1, N_wl)) * 1.0e-13,
        np.ones((N_layers, 1, 1, N_wl)) * 1.0e-14,
        np.ones((N_layers, 1, 1, 1, N_wl)) * 1.0e-14,
    )
    s_passthrough = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, kappa_contributions=contributions
    )
    atmosphere["kappa_cloud_eddysed"] = atmosphere["kappa_cloud_eddysed"] * 1.0e6
    s_modified = j_compute_spectrum(
        planet, star, model, atmosphere, opac, wl, kappa_contributions=contributions
    )
    np.testing.assert_array_equal(s_passthrough, s_modified)
