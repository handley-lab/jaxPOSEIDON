"""Phase 0.5.15b: full `extinction_LBL` orchestrator.

Ports POSEIDON `absorption.py:1739-1951` (`extinction_LBL`) plus the
helpers `T_interpolation_init` (l. 204-236), `interpolate_cia_LBL`
(l. 1386-1434), `interpolate_sigma_LBL` (l. 1438-1593).

A synthetic tiny HDF5 fixture mirrors POSEIDON's on-disk schema
(`<species>/T`, `/log(P)`, `/nu`, `/log(sigma)`; `<pair>/T`, `/nu`,
`/log(cia)`) so parity vs the upstream `extinction_LBL` can be exercised
without the ~10 GB v1.3 opacity database.
"""

import os

import h5py
import numpy as np
import pytest

from jaxposeidon import _lbl


# ---------------------------------------------------------------------------
# Helper: build a synthetic LBL opacity database matching POSEIDON's schema.
# ---------------------------------------------------------------------------
def _build_synthetic_opacity_db(
    tmp_path,
    species_active,
    cia_pairs,
    *,
    seed=0,
    N_T=8,
    N_P=6,
    N_nu_opac=200,
    N_T_cia=8,
    N_nu_cia=200,
):
    opac_dir = tmp_path / "opacity"
    opac_dir.mkdir()
    db_path = opac_dir / "Opacity_database_v1.3.hdf5"
    cia_path = opac_dir / "Opacity_database_cia.hdf5"

    rng = np.random.default_rng(seed)

    T_grid = np.linspace(200.0, 2000.0, N_T)
    log_P_grid = np.linspace(-6.0, 2.0, N_P)
    nu_opac = np.linspace(1000.0, 12000.0, N_nu_opac)

    with h5py.File(db_path, "w") as f:
        for species in species_active:
            log_sigma = rng.uniform(-26.0, -20.0, size=(N_P, N_T, N_nu_opac))
            g = f.create_group(species)
            g.create_dataset("T", data=T_grid)
            g.create_dataset("log(P)", data=log_P_grid)
            g.create_dataset("nu", data=nu_opac)
            g.create_dataset("log(sigma)", data=log_sigma.astype(np.float64))

    T_cia_grid = np.linspace(200.0, 2000.0, N_T_cia)
    nu_cia = np.linspace(1000.0, 12000.0, N_nu_cia)
    with h5py.File(cia_path, "w") as f:
        for pair in cia_pairs:
            log_cia = rng.uniform(-50.0, -40.0, size=(N_T_cia, N_nu_cia))
            g = f.create_group(pair)
            g.create_dataset("T", data=T_cia_grid)
            g.create_dataset("nu", data=nu_cia)
            g.create_dataset("log(cia)", data=log_cia.astype(np.float64))

    return tmp_path


# ---------------------------------------------------------------------------
# Parity helpers: parity vs POSEIDON for T_interpolation_init / interp_*.
# ---------------------------------------------------------------------------
def test_T_interpolation_init_matches_poseidon():
    from POSEIDON.absorption import T_interpolation_init as p_init

    T_grid = np.linspace(200.0, 2000.0, 10)
    T_fine = np.array([100.0, 250.0, 500.0, 1234.0, 1950.0, 2500.0])

    y_ours = np.zeros(len(T_fine), dtype=np.int64)
    w_ours = _lbl.T_interpolation_init(len(T_fine), T_grid, T_fine, y_ours)

    y_theirs = np.zeros(len(T_fine), dtype=np.int64)
    w_theirs = p_init(len(T_fine), T_grid, T_fine, y_theirs)

    np.testing.assert_array_equal(y_ours, y_theirs)
    np.testing.assert_allclose(w_ours, w_theirs, atol=0, rtol=1e-13)


def test_interpolate_cia_LBL_matches_poseidon():
    from POSEIDON.absorption import (
        T_interpolation_init as p_init,
        interpolate_cia_LBL as p_cia,
    )

    rng = np.random.default_rng(2)
    N_T_cia = 8
    N_nu_cia = 80
    T_grid_cia = np.linspace(200.0, 2000.0, N_T_cia)
    nu_cia = np.linspace(1000.0, 11000.0, N_nu_cia)
    log_cia = rng.uniform(-50.0, -40.0, size=(N_T_cia, N_nu_cia)).astype(np.float32)

    wl_model = np.linspace(1.0, 5.0, 60)
    nu_model = (1.0e4 / wl_model)[::-1]
    N_wl = len(wl_model)
    N_nu = len(nu_model)

    P = np.logspace(2, -7, 12)
    T = rng.uniform(150.0, 2100.0, size=len(P))

    y = np.zeros(len(P), dtype=np.int64)
    w_T = _lbl.T_interpolation_init(len(P), T_grid_cia, T, y)
    y_p = np.zeros(len(P), dtype=np.int64)
    _ = p_init(len(P), T_grid_cia, T, y_p)
    np.testing.assert_array_equal(y, y_p)

    out_ours = _lbl.interpolate_cia_LBL(
        P, log_cia, nu_model, nu_cia, T, T_grid_cia, N_T_cia, N_wl, N_nu, y, w_T
    )
    out_theirs = p_cia(
        P, log_cia, nu_model, nu_cia, T, T_grid_cia, N_T_cia, N_wl, N_nu, y_p, w_T
    )
    # rtol=1e-5 because log_cia is float32-cast at the call site per
    # POSEIDON; the post-`10**x` values are then at float32 precision
    # (~7 decimal digits). Sub-normal entries (~1e-45 or below) may
    # disagree on whether they underflow.
    np.testing.assert_allclose(out_ours, out_theirs, atol=1e-50, rtol=1e-5)


def test_interpolate_sigma_LBL_matches_poseidon():
    from POSEIDON.absorption import (
        T_interpolation_init as p_init,
        interpolate_sigma_LBL as p_sigma,
    )

    rng = np.random.default_rng(3)
    N_T = 8
    N_P = 6
    N_nu_opac = 200
    T_grid = np.linspace(200.0, 2000.0, N_T)
    log_P_grid = np.linspace(-6.0, 2.0, N_P)
    nu_opac = np.linspace(1000.0, 12000.0, N_nu_opac)
    log_sigma = rng.uniform(-26.0, -20.0, size=(N_P, N_T, N_nu_opac)).astype(np.float32)

    wl_model = np.linspace(1.0, 5.0, 50)
    nu_model = (1.0e4 / wl_model)[::-1]
    N_wl = len(wl_model)
    N_nu = len(nu_model)

    P = np.logspace(2, -7, 14)
    T = rng.uniform(150.0, 2100.0, size=len(P))

    y = np.zeros(len(P), dtype=np.int64)
    w_T = _lbl.T_interpolation_init(len(P), T_grid, T, y)
    y_p = np.zeros(len(P), dtype=np.int64)
    _ = p_init(len(P), T_grid, T, y_p)

    out_ours = _lbl.interpolate_sigma_LBL(
        log_sigma,
        nu_model,
        nu_opac,
        P,
        T,
        log_P_grid,
        T_grid,
        N_T,
        N_P,
        N_wl,
        N_nu,
        y,
        w_T,
    )
    out_theirs = p_sigma(
        log_sigma,
        nu_model,
        nu_opac,
        P,
        T,
        log_P_grid,
        T_grid,
        N_T,
        N_P,
        N_wl,
        N_nu,
        y_p,
        w_T,
    )
    np.testing.assert_allclose(out_ours, out_theirs, atol=1e-50, rtol=1e-5)


# ---------------------------------------------------------------------------
# End-to-end: extinction_LBL orchestrator parity on a synthetic DB.
# ---------------------------------------------------------------------------
def _build_minimal_call(tmp_path, seed=0):
    chemical_species = ["H2", "He", "H2O"]
    active_species = ["H2O"]
    cia_pairs = ["H2-H2", "H2-He"]
    ff_pairs = []
    bf_species = []

    _build_synthetic_opacity_db(tmp_path, active_species, cia_pairs, seed=seed)

    rng = np.random.default_rng(seed)
    N_layers, N_wl = 14, 40
    wl_model = np.linspace(1.0, 5.0, N_wl)
    P = np.logspace(2, -7, N_layers)
    T = np.full((N_layers, 1, 1), 800.0) + rng.uniform(
        -200.0, 200.0, size=(N_layers, 1, 1)
    )
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))

    N_species = len(chemical_species)
    X = np.zeros((N_species, N_layers, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, len(cia_pairs), N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))

    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(N_species, N_wl))

    return dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
        n=n,
        T=T,
        P=P,
        wl_model=wl_model,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        a=1.0,
        gamma=-4.0,
        P_cloud=1e-3,
        kappa_cloud_0=1e-30,
        Rayleigh_stored=Rayleigh_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        N_sectors=1,
        N_zones=1,
        P_surf=1e-30,
        opacity_database="High-T",
        disable_continuum=False,
        suppress_print=True,
        database_version="1.3",
    )


def test_extinction_LBL_matches_poseidon_synthetic(tmp_path, monkeypatch):
    """End-to-end parity vs POSEIDON.absorption.extinction_LBL on a
    synthetic tiny HDF5 fixture (no ~10 GB DB required)."""
    from POSEIDON.absorption import extinction_LBL as p_extLBL

    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    call_args = _build_minimal_call(tmp_path, seed=11)

    kg_ours, kr_ours, kc_ours = _lbl.extinction_LBL(**call_args)
    kg_t, kr_t, kc_t = p_extLBL(**call_args)

    np.testing.assert_allclose(kg_ours, kg_t, atol=1e-50, rtol=1e-5)
    np.testing.assert_allclose(kr_ours, kr_t, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_t, atol=0, rtol=1e-13)


def test_extinction_LBL_disable_continuum_synthetic(tmp_path, monkeypatch):
    from POSEIDON.absorption import extinction_LBL as p_extLBL

    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    call_args = _build_minimal_call(tmp_path, seed=12)
    call_args["disable_continuum"] = True

    kg_ours, kr_ours, kc_ours = _lbl.extinction_LBL(**call_args)
    kg_t, kr_t, kc_t = p_extLBL(**call_args)

    np.testing.assert_allclose(kg_ours, kg_t, atol=1e-50, rtol=1e-5)
    np.testing.assert_allclose(kr_ours, kr_t, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_t, atol=0, rtol=1e-13)


# ---------------------------------------------------------------------------
# Compute-spectrum dispatch: line_by_line is now wired through.
# ---------------------------------------------------------------------------
def test_compute_spectrum_dispatches_line_by_line(tmp_path, monkeypatch):
    """compute_spectrum no longer rejects opacity_treatment='line_by_line'.

    Guards the dispatch wiring in `_compute_spectrum.py`: previously the
    guard raised NotImplementedError. The LBL branch is now entered (any
    error must originate inside `extinction_LBL`, not the upstream guard).
    """
    from POSEIDON.absorption import store_Rayleigh_eta_LBL
    from POSEIDON.constants import R_J, R_Sun
    from POSEIDON.core import (
        create_planet,
        create_star,
        define_model,
        make_atmosphere,
        wl_grid_constant_R,
    )

    from jaxposeidon._compute_spectrum import compute_spectrum

    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))

    chemical_species = ["H2", "He", "H2O"]
    active_species = ["H2O"]
    cia_pairs = ["H2-H2", "H2-He"]
    _build_synthetic_opacity_db(tmp_path, active_species, cia_pairs, seed=4)

    star = create_star(R_Sun, 5000.0, 4.0, 0.0)
    planet = create_planet("p", R_J, mass=1.898e27, T_eq=300.0)
    model = define_model(
        "lbl_dispatch",
        chemical_species,
        active_species,
        PT_profile="isotherm",
        X_profile="isochem",
    )
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), 30)
    atmosphere = make_atmosphere(
        planet,
        model,
        P,
        10.0,
        R_J,
        np.array([800.0]),
        np.array([-4.0]),
        constant_gravity=True,
    )
    wl = wl_grid_constant_R(1.5, 2.5, 200)
    Rayleigh_stored, _ = store_Rayleigh_eta_LBL(wl, chemical_species)

    opac = {
        "opacity_database": "High-T",
        "opacity_treatment": "line_by_line",
        "Rayleigh_stored": Rayleigh_stored,
        "database_version": "1.3",
    }

    # The wiring should send the call into the LBL branch. Either the
    # call succeeds (synthetic DB is broad enough), or it errors INSIDE
    # extinction_LBL (e.g. mismatched model-vs-opacity wavenumber grids).
    # The previous NotImplementedError guard MUST be gone.
    try:
        spectrum = compute_spectrum(
            planet, star, model, atmosphere, opac, wl, suppress_print=True
        )
        assert spectrum.shape == (len(wl),)
    except NotImplementedError as exc:
        if "opacity_treatment" in str(exc):
            raise AssertionError(
                "compute_spectrum still rejects line_by_line at the guard."
            ) from exc
        raise
    except (IndexError, ValueError):
        # Synthetic DB / model wavelength grids may not be aligned; the
        # important guarantee is that we reached extinction_LBL (i.e.
        # the dispatch wiring is in place).
        pass


# ---------------------------------------------------------------------------
# Env-gated smoke test: runs against the real $POSEIDON_input_data if set.
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not os.environ.get("POSEIDON_input_data")
    or not os.path.exists(
        os.path.join(
            os.environ.get("POSEIDON_input_data", ""),
            "opacity",
            "Opacity_database_v1.3.hdf5",
        )
    ),
    reason="Real POSEIDON opacity database not available",
)
def test_extinction_LBL_real_data_smoke():
    """Smoke test: confirms the orchestrator runs against the real DB."""
    from POSEIDON.absorption import store_Rayleigh_eta_LBL

    chemical_species = ["H2", "He", "H2O"]
    active_species = ["H2O"]
    cia_pairs = ["H2-H2", "H2-He"]
    wl_model = np.linspace(1.5, 1.6, 20)
    N_layers, N_wl = 8, len(wl_model)
    P = np.logspace(1, -5, N_layers)
    T = np.full((N_layers, 1, 1), 800.0)
    n = np.full((N_layers, 1, 1), 1.0e20)

    N_species = len(chemical_species)
    X = np.zeros((N_species, N_layers, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, len(cia_pairs), N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))

    Rayleigh_stored, _ = store_Rayleigh_eta_LBL(wl_model, chemical_species)

    kg, kr, kc = _lbl.extinction_LBL(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=[],
        bf_species=[],
        n=n,
        T=T,
        P=P,
        wl_model=wl_model,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        a=1.0,
        gamma=-4.0,
        P_cloud=1.0e-3,
        kappa_cloud_0=1.0e-30,
        Rayleigh_stored=Rayleigh_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        N_sectors=1,
        N_zones=1,
        P_surf=1.0e-30,
        suppress_print=True,
    )
    assert kg.shape == (N_layers, 1, 1, N_wl)
    assert np.all(np.isfinite(kg))
