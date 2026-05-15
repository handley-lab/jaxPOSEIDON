"""Phase 0.5.15c: multi-(sector, zone) `extinction_LBL` orchestrator.

Lifts the `N_sectors == N_zones == 1` restriction from Phase 0.5.15b.
POSEIDON's `absorption.py:1739-1951` iterates over `(j, k)` and calls
`compute_kappa_LBL` once per pair; this test exercises the orchestrator
across `(N_sectors, N_zones) in {(1,1), (2,1), (1,2), (2,2)}` and checks
parity with the upstream `extinction_LBL`.

The synthetic HDF5 fixture matches the one in
`test_phase_v05_15b_extinction_lbl.py` so the two phases share the same
opacity-database schema.
"""

import h5py
import numpy as np
import pytest

from jaxposeidon import _lbl


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


def _build_call_args(tmp_path, N_sectors, N_zones, seed=0):
    chemical_species = ["H2", "He", "H2O"]
    active_species = ["H2O"]
    cia_pairs = ["H2-H2", "H2-He"]
    ff_pairs = []
    bf_species = []

    _build_synthetic_opacity_db(tmp_path, active_species, cia_pairs, seed=seed)

    rng = np.random.default_rng(seed)
    N_layers, N_wl = 12, 40
    wl_model = np.linspace(1.0, 5.0, N_wl)
    P = np.logspace(2, -7, N_layers)

    T = np.full((N_layers, N_sectors, N_zones), 800.0) + rng.uniform(
        -200.0, 200.0, size=(N_layers, N_sectors, N_zones)
    )
    n = rng.uniform(1e15, 1e25, size=(N_layers, N_sectors, N_zones))

    N_species = len(chemical_species)
    X = np.zeros((N_species, N_layers, N_sectors, N_zones))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, len(cia_pairs), N_layers, N_sectors, N_zones))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, N_sectors, N_zones))
    X_bf = np.zeros((0, N_layers, N_sectors, N_zones))

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
        N_sectors=N_sectors,
        N_zones=N_zones,
        P_surf=1e-30,
        opacity_database="High-T",
        disable_continuum=False,
        suppress_print=True,
        database_version="1.3",
    )


@pytest.mark.parametrize("N_sectors,N_zones", [(1, 1), (2, 1), (1, 2), (2, 2)])
def test_extinction_LBL_multi_sector_zone_parity(
    tmp_path, monkeypatch, N_sectors, N_zones
):
    """End-to-end parity vs POSEIDON.absorption.extinction_LBL across
    a grid of (N_sectors, N_zones) configurations.

    POSEIDON's upstream `extinction_LBL` closes the HDF5 file handles
    inside the `(j, k)` loop, so the upstream function only works for
    `(1, 1)`. To compare against a working oracle for `N_sectors > 1`
    or `N_zones > 1`, we re-run POSEIDON's path on each `(j, k)` slice
    individually (using a `1 x 1` view of the per-pair inputs) and
    assemble the expected 4-D arrays. For `(1, 1)` we still test
    against the unmodified POSEIDON call to lock the bit-equivalence
    baseline established in Phase 0.5.15b.
    """
    from POSEIDON.absorption import extinction_LBL as p_extLBL

    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    args = _build_call_args(
        tmp_path, N_sectors, N_zones, seed=44 + N_sectors * 5 + N_zones
    )

    kg_ours, kr_ours, kc_ours = _lbl.extinction_LBL(**args)

    assert kg_ours.shape == (
        len(args["P"]),
        N_sectors,
        N_zones,
        len(args["wl_model"]),
    )

    if N_sectors == 1 and N_zones == 1:
        kg_t, kr_t, kc_t = p_extLBL(**args)
        np.testing.assert_allclose(kg_ours, kg_t, atol=1e-50, rtol=1e-5)
        np.testing.assert_allclose(kr_ours, kr_t, atol=0, rtol=1e-13)
        np.testing.assert_allclose(kc_ours, kc_t, atol=0, rtol=1e-13)
        return

    kg_exp = np.zeros_like(kg_ours)
    kr_exp = np.zeros_like(kr_ours)
    kc_exp = np.zeros_like(kc_ours)

    for j in range(N_sectors):
        for k in range(N_zones):
            sub = dict(args)
            sub["N_sectors"] = 1
            sub["N_zones"] = 1
            sub["T"] = args["T"][:, j : j + 1, k : k + 1].copy()
            sub["n"] = args["n"][:, j : j + 1, k : k + 1].copy()
            sub["X"] = args["X"][:, :, j : j + 1, k : k + 1].copy()
            sub["X_active"] = args["X_active"][:, :, j : j + 1, k : k + 1].copy()
            sub["X_cia"] = args["X_cia"][:, :, :, j : j + 1, k : k + 1].copy()
            sub["X_ff"] = args["X_ff"][:, :, :, j : j + 1, k : k + 1].copy()
            sub["X_bf"] = args["X_bf"][:, :, j : j + 1, k : k + 1].copy()

            kg_jk, kr_jk, kc_jk = p_extLBL(**sub)
            kg_exp[:, j : j + 1, k : k + 1, :] = kg_jk
            kr_exp[:, j : j + 1, k : k + 1, :] = kr_jk
            kc_exp[:, j : j + 1, k : k + 1, :] = kc_jk

    np.testing.assert_allclose(kg_ours, kg_exp, atol=1e-50, rtol=1e-5)
    np.testing.assert_allclose(kr_ours, kr_exp, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_exp, atol=0, rtol=1e-13)


@pytest.mark.parametrize("N_sectors,N_zones", [(2, 1), (1, 2), (2, 2)])
def test_extinction_LBL_multi_dim_haze_deck_parity(
    tmp_path, monkeypatch, N_sectors, N_zones
):
    """Multi-(sector, zone) parity with haze + deck enabled, per-slice."""
    from POSEIDON.absorption import extinction_LBL as p_extLBL

    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    args = _build_call_args(tmp_path, N_sectors, N_zones, seed=77)
    args["enable_haze"] = 1
    args["enable_deck"] = 1

    kg_ours, kr_ours, kc_ours = _lbl.extinction_LBL(**args)

    kg_exp = np.zeros_like(kg_ours)
    kr_exp = np.zeros_like(kr_ours)
    kc_exp = np.zeros_like(kc_ours)

    for j in range(N_sectors):
        for k in range(N_zones):
            sub = dict(args)
            sub["N_sectors"] = 1
            sub["N_zones"] = 1
            sub["T"] = args["T"][:, j : j + 1, k : k + 1].copy()
            sub["n"] = args["n"][:, j : j + 1, k : k + 1].copy()
            sub["X"] = args["X"][:, :, j : j + 1, k : k + 1].copy()
            sub["X_active"] = args["X_active"][:, :, j : j + 1, k : k + 1].copy()
            sub["X_cia"] = args["X_cia"][:, :, :, j : j + 1, k : k + 1].copy()
            sub["X_ff"] = args["X_ff"][:, :, :, j : j + 1, k : k + 1].copy()
            sub["X_bf"] = args["X_bf"][:, :, j : j + 1, k : k + 1].copy()

            kg_jk, kr_jk, kc_jk = p_extLBL(**sub)
            kg_exp[:, j : j + 1, k : k + 1, :] = kg_jk
            kr_exp[:, j : j + 1, k : k + 1, :] = kr_jk
            kc_exp[:, j : j + 1, k : k + 1, :] = kc_jk

    np.testing.assert_allclose(kg_ours, kg_exp, atol=1e-50, rtol=1e-5)
    np.testing.assert_allclose(kr_ours, kr_exp, atol=0, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_exp, atol=0, rtol=1e-13)
    assert np.any(kc_ours > 0.0)
