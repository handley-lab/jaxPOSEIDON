"""Phase 0.5.15 (subset): line-by-line extinction kernel + table loader.

Ports POSEIDON `absorption.py:1627-1736` (`compute_kappa_LBL`). The
full `extinction_LBL` orchestrator (which calls this kernel after
interpolating cross-sections from ~10 GB HDF5 grids) requires real
opacity data and is a follow-up.
"""

import os

import numpy as np
import pytest

from jaxposeidon import _lbl, _lbl_table_loader


def test_compute_kappa_LBL_matches_poseidon():
    """LBL kernel parity vs POSEIDON.absorption.compute_kappa_LBL."""
    from POSEIDON.absorption import compute_kappa_LBL as p_kappa

    rng = np.random.default_rng(0)
    N_layers, N_wl = 20, 30
    N_species = 3  # H2, He, H2O
    N_species_active = 1  # H2O
    N_cia_pairs = 2  # H2-H2, H2-He
    N_ff_pairs = 0
    N_bf_species = 0

    wl_model = np.linspace(1.0, 5.0, N_wl)
    P = np.logspace(2, -7, N_layers)
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))
    X = np.zeros((N_species, N_layers, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, N_cia_pairs, N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))

    sigma_interp = rng.uniform(0.0, 1e-22, size=(N_species_active, N_layers, N_wl))
    cia_interp = rng.uniform(0.0, 1e-44, size=(N_cia_pairs, N_layers, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(N_species, N_wl))
    ff_stored = np.zeros((0, N_layers, N_wl))
    bf_stored = np.zeros((0, N_wl))

    args_common = dict(
        wl_model=wl_model,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        n=n,
        P=P,
        a=1.0,
        gamma=-4.0,
        P_cloud=1e-3,
        kappa_cloud_0=1e-30,
        N_species=N_species,
        N_species_active=N_species_active,
        N_cia_pairs=N_cia_pairs,
        N_ff_pairs=N_ff_pairs,
        N_bf_species=N_bf_species,
        sigma_interp=sigma_interp,
        cia_interp=cia_interp,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=ff_stored,
        bf_stored=bf_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        P_surf=1e-30,
        disable_continuum=False,
    )

    kg_ours = np.zeros((N_layers, 1, 1, N_wl))
    kr_ours = np.zeros((N_layers, 1, 1, N_wl))
    kc_ours = np.zeros((N_layers, 1, 1, N_wl))
    _lbl.compute_kappa_LBL(
        j=0,
        k=0,
        kappa_gas=kg_ours,
        kappa_Ray=kr_ours,
        kappa_cloud=kc_ours,
        **args_common,
    )

    kg_theirs = np.zeros((N_layers, 1, 1, N_wl))
    kr_theirs = np.zeros((N_layers, 1, 1, N_wl))
    kc_theirs = np.zeros((N_layers, 1, 1, N_wl))
    p_kappa(
        0,
        0,
        wl_model=wl_model,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        n=n,
        P=P,
        a=1.0,
        gamma=-4.0,
        P_cloud=1e-3,
        kappa_cloud_0=1e-30,
        N_species=N_species,
        N_species_active=N_species_active,
        N_cia_pairs=N_cia_pairs,
        N_ff_pairs=N_ff_pairs,
        N_bf_species=N_bf_species,
        sigma_interp=sigma_interp,
        cia_interp=cia_interp,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=ff_stored,
        bf_stored=bf_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        kappa_gas=kg_theirs,
        kappa_Ray=kr_theirs,
        kappa_cloud=kc_theirs,
        P_surf=1e-30,
        disable_continuum=False,
    )

    np.testing.assert_allclose(kg_ours, kg_theirs, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kr_ours, kr_theirs, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_theirs, atol=0, rtol=1e-13)


def test_compute_kappa_LBL_disable_continuum_matches_poseidon():
    """disable_continuum=True parity vs POSEIDON compute_kappa_LBL."""
    from POSEIDON.absorption import compute_kappa_LBL as p_kappa

    rng = np.random.default_rng(1)
    N_layers, N_wl = 15, 25
    args = dict(
        wl_model=np.linspace(1.0, 5.0, N_wl),
        X=np.full((2, N_layers, 1, 1), 0.4),
        X_active=np.full((1, N_layers, 1, 1), 0.001),
        X_cia=np.full((2, 1, N_layers, 1, 1), 0.5),
        X_ff=np.zeros((2, 0, N_layers, 1, 1)),
        X_bf=np.zeros((0, N_layers, 1, 1)),
        n=np.full((N_layers, 1, 1), 1e20),
        P=np.logspace(1, -5, N_layers),
        a=1.0,
        gamma=-4.0,
        P_cloud=1e-3,
        kappa_cloud_0=1e-30,
        N_species=2,
        N_species_active=1,
        N_cia_pairs=1,
        N_ff_pairs=0,
        N_bf_species=0,
        sigma_interp=rng.uniform(1e-23, 1e-22, size=(1, N_layers, N_wl)),
        cia_interp=rng.uniform(1e-45, 1e-44, size=(1, N_layers, N_wl)),
        Rayleigh_stored=rng.uniform(1e-28, 1e-27, size=(2, N_wl)),
        ff_stored=np.zeros((0, N_layers, N_wl)),
        bf_stored=np.zeros((0, N_wl)),
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        P_surf=1e-30,
        disable_continuum=True,
    )
    kg_o = np.zeros((N_layers, 1, 1, N_wl))
    kr_o = np.zeros((N_layers, 1, 1, N_wl))
    kc_o = np.zeros((N_layers, 1, 1, N_wl))
    _lbl.compute_kappa_LBL(
        j=0,
        k=0,
        kappa_gas=kg_o,
        kappa_Ray=kr_o,
        kappa_cloud=kc_o,
        **args,
    )
    kg_t = np.zeros((N_layers, 1, 1, N_wl))
    kr_t = np.zeros((N_layers, 1, 1, N_wl))
    kc_t = np.zeros((N_layers, 1, 1, N_wl))
    p_kappa(
        0,
        0,
        kappa_gas=kg_t,
        kappa_Ray=kr_t,
        kappa_cloud=kc_t,
        **args,
    )
    np.testing.assert_allclose(kg_o, kg_t, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kr_o, kr_t, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_o, kc_t, atol=0, rtol=1e-13)


def test_compute_kappa_LBL_disable_continuum_zeroes_Rayleigh():
    """disable_continuum zeroes Rayleigh in the LBL kernel (behavioral)."""
    rng = np.random.default_rng(0)
    N_layers, N_wl = 10, 10
    args = dict(
        wl_model=np.linspace(1.0, 5.0, N_wl),
        X=np.full((1, N_layers, 1, 1), 0.001),
        X_active=np.full((1, N_layers, 1, 1), 0.001),
        X_cia=np.full((2, 1, N_layers, 1, 1), 0.5),
        X_ff=np.zeros((2, 0, N_layers, 1, 1)),
        X_bf=np.zeros((0, N_layers, 1, 1)),
        n=np.full((N_layers, 1, 1), 1e20),
        P=np.logspace(1, -5, N_layers),
        a=1.0,
        gamma=-4.0,
        P_cloud=1e-3,
        kappa_cloud_0=1e-30,
        N_species=1,
        N_species_active=1,
        N_cia_pairs=1,
        N_ff_pairs=0,
        N_bf_species=0,
        sigma_interp=rng.uniform(1e-23, 1e-22, size=(1, N_layers, N_wl)),
        cia_interp=rng.uniform(1e-45, 1e-44, size=(1, N_layers, N_wl)),
        Rayleigh_stored=rng.uniform(1e-28, 1e-27, size=(1, N_wl)),
        ff_stored=np.zeros((0, N_layers, N_wl)),
        bf_stored=np.zeros((0, N_wl)),
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        P_surf=1e-30,
    )
    kg = np.zeros((N_layers, 1, 1, N_wl))
    kr = np.zeros((N_layers, 1, 1, N_wl))
    kc = np.zeros((N_layers, 1, 1, N_wl))
    _lbl.compute_kappa_LBL(
        j=0,
        k=0,
        kappa_gas=kg,
        kappa_Ray=kr,
        kappa_cloud=kc,
        disable_continuum=True,
        **args,
    )
    # Rayleigh entirely zero
    np.testing.assert_array_equal(kr, np.zeros_like(kr))
    # kappa_gas still nonzero (active opacity remains)
    assert np.any(kg > 0.0)


def test_open_opacity_files_requires_env_var(monkeypatch):
    monkeypatch.delenv("POSEIDON_input_data", raising=False)
    with pytest.raises(
        Exception, match="POSEIDON cannot locate the opacity input data"
    ):
        _lbl_table_loader.open_opacity_files()


def test_open_opacity_files_path_construction(tmp_path, monkeypatch):
    """Path construction parity vs POSEIDON's exact string concatenation
    at absorption.py:1799-1818."""
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    # Build empty HDF5s at all the expected paths so h5py.File succeeds.
    import h5py

    (tmp_path / "opacity").mkdir()
    for name in (
        "Opacity_database_v1.3.hdf5",
        "Opacity_database_v1.2.hdf5",
        "Opacity_database_v1.0.hdf5",
        "Opacity_database_0.01cm-1_Temperate.hdf5",
        "Opacity_database_cia.hdf5",
    ):
        h5py.File(tmp_path / "opacity" / name, "w").close()

    expectations = [
        ("High-T", "1.3", "Opacity_database_v1.3.hdf5"),
        ("High-T", "1.2", "Opacity_database_v1.2.hdf5"),
        ("High-T", "1.0", "Opacity_database_v1.0.hdf5"),
        ("Temperate", "1.3", "Opacity_database_0.01cm-1_Temperate.hdf5"),
    ]
    for db, ver, expected_name in expectations:
        opac, cia = _lbl_table_loader.open_opacity_files(db, ver)
        assert os.path.basename(opac.filename) == expected_name
        assert os.path.basename(cia.filename) == "Opacity_database_cia.hdf5"
        opac.close()
        cia.close()


def test_open_opacity_files_rejects_unknown_database_version(monkeypatch, tmp_path):
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    with pytest.raises(Exception, match="Invalid opacity database version"):
        _lbl_table_loader.open_opacity_files(database_version="bogus")
