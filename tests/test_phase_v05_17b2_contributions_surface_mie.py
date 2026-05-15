"""Phase 0.5.17b2 parity tests for surface + Mie branches of the
contribution kernels.

Mirrors ``POSEIDON.contributions.extinction_spectral_contribution``
(POSEIDON ``contributions.py:414-470``) and
``extinction_pressure_contribution`` (``contributions.py:1433-1548``)
for ``enable_surface=1`` / ``enable_Mie=1``, with both the
"no opaque deck" (``len(n_aerosol_array) == len(sigma_Mie_array)``)
and "opaque-deck-first-element"
(``len(n_aerosol_array) == len(sigma_Mie_array) + 1``) layouts, and
across ``cloud_contribution`` / ``cloud_total_contribution`` /
``aerosol_species_index`` selectors.
"""

import numpy as np
import pytest

from jaxposeidon._contributions import (
    extinction_pressure_contribution,
    extinction_spectral_contribution,
)


def _setup(
    N_layers=14, N_wl=8, N_T_fine=5, N_P_fine=6, N_aerosol=2, opaque=False, seed=0
):
    rng = np.random.default_rng(seed)
    chemical_species = np.array(["H2", "He", "H2O", "CH4"])
    active_species = np.array(["H2O", "CH4"])
    cia_pairs = np.array(["H2-H2", "H2-He"])
    ff_pairs = np.array([], dtype=str)
    bf_species = np.array([], dtype=str)
    aerosol_species = np.array(["MgSiO3", "Fe"])[:N_aerosol]

    P = np.logspace(np.log10(100.0), np.log10(1.0e-6), N_layers)
    T = 1000.0 + 200.0 * rng.standard_normal((N_layers, 1, 1))
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))
    wl = np.linspace(1.0, 5.0, N_wl)

    X = np.zeros((4, N_layers, 1, 1))
    X[0] = 0.84
    X[1] = 0.149
    X[2] = 0.005
    X[3] = 0.006

    X_active = X[2:4]

    X_cia = np.zeros((2, 2, N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]

    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))

    T_fine = np.linspace(500.0, 2000.0, N_T_fine)
    log_P_fine = np.linspace(-6.0, 2.0, N_P_fine)

    sigma_stored = rng.uniform(0.0, 1e-22, size=(2, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(0.0, 1e-44, size=(2, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(4, N_wl))
    ff_stored = np.zeros((0, N_T_fine, N_wl))
    bf_stored = np.zeros((0, N_wl))

    # Opaque-deck-first-element layout: n_aerosol_array has one extra
    # element (the deck) at index 0; sigma_Mie_array is aligned with
    # aerosol_species starting at index aerosol-1.
    if opaque:
        n_aerosol_array = rng.uniform(1.0, 1e3, size=(N_aerosol + 1, N_layers, 1, 1))
        sigma_Mie_array = rng.uniform(0.0, 1e-20, size=(N_aerosol, N_wl))
    else:
        n_aerosol_array = rng.uniform(1.0, 1e3, size=(N_aerosol, N_layers, 1, 1))
        sigma_Mie_array = rng.uniform(0.0, 1e-20, size=(N_aerosol, N_wl))

    return dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
        aerosol_species=aerosol_species,
        n=n,
        T=T,
        P=P,
        wl=wl,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        a=1.0,
        gamma=-4.0,
        P_cloud=np.array([1.0e-3]),
        kappa_cloud_0=1.0e-30,
        sigma_stored=sigma_stored,
        cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=ff_stored,
        bf_stored=bf_stored,
        enable_haze=0,
        enable_deck=0,
        enable_surface=0,
        N_sectors=1,
        N_zones=1,
        T_fine=T_fine,
        log_P_fine=log_P_fine,
        P_surf=1.0e-2,
        enable_Mie=0,
        n_aerosol_array=n_aerosol_array,
        sigma_Mie_array=sigma_Mie_array,
        P_deep=1000.0,
    )


def _call_poseidon_spec(cfg, **opts):
    from POSEIDON.contributions import extinction_spectral_contribution as p_fn

    return p_fn(
        cfg["chemical_species"],
        cfg["active_species"],
        cfg["cia_pairs"],
        cfg["ff_pairs"],
        cfg["bf_species"],
        cfg["aerosol_species"],
        cfg["n"],
        cfg["T"],
        cfg["P"],
        cfg["wl"],
        cfg["X"],
        cfg["X_active"],
        cfg["X_cia"],
        cfg["X_ff"],
        cfg["X_bf"],
        cfg["a"],
        cfg["gamma"],
        cfg["P_cloud"],
        cfg["kappa_cloud_0"],
        cfg["sigma_stored"],
        cfg["cia_stored"],
        cfg["Rayleigh_stored"],
        cfg["ff_stored"],
        cfg["bf_stored"],
        cfg["enable_haze"],
        cfg["enable_deck"],
        cfg["enable_surface"],
        cfg["N_sectors"],
        cfg["N_zones"],
        cfg["T_fine"],
        cfg["log_P_fine"],
        cfg["P_surf"],
        cfg["enable_Mie"],
        cfg["n_aerosol_array"],
        cfg["sigma_Mie_array"],
        cfg["P_deep"],
        opts.get("contribution_species", ""),
        opts.get("bulk_species", False),
        opts.get("cloud_contribution", False),
        opts.get("cloud_species", ""),
        opts.get("cloud_total_contribution", False),
    )


def _call_poseidon_press(cfg, **opts):
    from POSEIDON.contributions import extinction_pressure_contribution as p_fn

    return p_fn(
        cfg["chemical_species"],
        cfg["active_species"],
        cfg["cia_pairs"],
        cfg["ff_pairs"],
        cfg["bf_species"],
        cfg["aerosol_species"],
        cfg["n"],
        cfg["T"],
        cfg["P"],
        cfg["wl"],
        cfg["X"],
        cfg["X_active"],
        cfg["X_cia"],
        cfg["X_ff"],
        cfg["X_bf"],
        cfg["a"],
        cfg["gamma"],
        cfg["P_cloud"],
        cfg["kappa_cloud_0"],
        cfg["sigma_stored"],
        cfg["cia_stored"],
        cfg["Rayleigh_stored"],
        cfg["ff_stored"],
        cfg["bf_stored"],
        cfg["enable_haze"],
        cfg["enable_deck"],
        cfg["enable_surface"],
        cfg["N_sectors"],
        cfg["N_zones"],
        cfg["T_fine"],
        cfg["log_P_fine"],
        cfg["P_surf"],
        cfg["enable_Mie"],
        cfg["n_aerosol_array"],
        cfg["sigma_Mie_array"],
        cfg["P_deep"],
        opts.get("contribution_species", ""),
        opts.get("bulk_species", False),
        opts.get("cloud_contribution", False),
        opts.get("cloud_species", ""),
        opts.get("cloud_total_contribution", False),
        opts.get("layer_to_ignore", 0),
        opts.get("total_pressure_contribution", False),
    )


def _assert_close(a, b):
    np.testing.assert_allclose(a, b, atol=1e-22, rtol=1e-13)


# ---- surface-only ------------------------------------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O"},
        {"contribution_species": "H2", "bulk_species": True},
        {"cloud_contribution": True},
    ],
    ids=["H2O", "bulk", "cloud"],
)
def test_spectral_surface_matches_poseidon(opts):
    cfg = _setup()
    cfg["enable_surface"] = 1
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O", "layer_to_ignore": 5},
        {"total_pressure_contribution": True, "layer_to_ignore": 6},
    ],
    ids=["H2O", "total"],
)
def test_pressure_surface_matches_poseidon(opts):
    cfg = _setup()
    cfg["enable_surface"] = 1
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- Mie: no-opaque-deck layout ---------------------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O"},
        {"cloud_contribution": True, "cloud_species": "MgSiO3"},
        {"cloud_contribution": True, "cloud_species": "Fe"},
        {"cloud_contribution": True, "cloud_total_contribution": True},
    ],
    ids=["gas-H2O", "cloud-MgSiO3", "cloud-Fe", "cloud-total"],
)
def test_spectral_Mie_no_opaque_matches_poseidon(opts):
    cfg = _setup(opaque=False)
    cfg["enable_Mie"] = 1
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O", "layer_to_ignore": 5},
        {
            "cloud_contribution": True,
            "cloud_species": "MgSiO3",
            "layer_to_ignore": 7,
        },
        {
            "cloud_contribution": True,
            "cloud_total_contribution": True,
            "layer_to_ignore": 8,
        },
        {"total_pressure_contribution": True, "layer_to_ignore": 6},
    ],
    ids=["gas-H2O", "cloud-MgSiO3", "cloud-total", "total"],
)
def test_pressure_Mie_no_opaque_matches_poseidon(opts):
    cfg = _setup(opaque=False)
    cfg["enable_Mie"] = 1
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- Mie: opaque-deck-first-element layout ----------------------------------


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O"},
        {"cloud_contribution": True, "cloud_species": "MgSiO3"},
        {"cloud_contribution": True, "cloud_species": "Fe"},
        {"cloud_contribution": True, "cloud_total_contribution": True},
    ],
    ids=["gas-H2O", "cloud-MgSiO3", "cloud-Fe", "cloud-total"],
)
def test_spectral_Mie_opaque_matches_poseidon(opts):
    cfg = _setup(opaque=True)
    cfg["enable_Mie"] = 1
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O", "layer_to_ignore": 5},
        {
            "cloud_contribution": True,
            "cloud_species": "MgSiO3",
            "layer_to_ignore": 7,
        },
        {
            "cloud_contribution": True,
            "cloud_total_contribution": True,
            "layer_to_ignore": 8,
        },
        {"total_pressure_contribution": True, "layer_to_ignore": 6},
    ],
    ids=["gas-H2O", "cloud-MgSiO3", "cloud-total", "total"],
)
def test_pressure_Mie_opaque_matches_poseidon(opts):
    cfg = _setup(opaque=True)
    cfg["enable_Mie"] = 1
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- surface + Mie combined --------------------------------------------------


def test_spectral_surface_and_Mie_matches_poseidon():
    cfg = _setup(opaque=False)
    cfg["enable_surface"] = 1
    cfg["enable_Mie"] = 1
    cfg["enable_haze"] = 1
    cfg["enable_deck"] = 1
    opts = {"cloud_contribution": True, "cloud_species": "MgSiO3"}
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


def test_pressure_surface_and_Mie_matches_poseidon():
    cfg = _setup(opaque=True)
    cfg["enable_surface"] = 1
    cfg["enable_Mie"] = 1
    cfg["enable_haze"] = 1
    cfg["enable_deck"] = 1
    opts = {"total_pressure_contribution": True, "layer_to_ignore": 6}
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)
