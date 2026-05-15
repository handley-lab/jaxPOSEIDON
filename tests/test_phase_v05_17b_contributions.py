"""Phase 0.5.17b parity tests against POSEIDON's contribution kernels.

Mirrors ``POSEIDON.contributions.extinction_spectral_contribution`` and
``POSEIDON.contributions.extinction_pressure_contribution`` over the v0
envelope (no surface, no Mie), exercising every selector path:
per-species, bulk-only, cloud_contribution, total_pressure_contribution,
H-minus ff/bf, and multi-sector / multi-zone parity.
"""

import numpy as np
import pytest

from jaxposeidon._contributions import (
    extinction_pressure_contribution,
    extinction_spectral_contribution,
)


def _setup(N_layers=18, N_wl=12, N_T_fine=5, N_P_fine=6, seed=0):
    rng = np.random.default_rng(seed)
    chemical_species = np.array(["H2", "He", "H2O", "CH4"])
    active_species = np.array(["H2O", "CH4"])
    cia_pairs = np.array(["H2-H2", "H2-He"])
    ff_pairs = np.array([], dtype=str)
    bf_species = np.array([], dtype=str)
    aerosol_species = np.array([], dtype=str)

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
        P_surf=1.0e-30,
        enable_Mie=0,
        n_aerosol_array=np.zeros((0, N_layers, 1, 1)),
        sigma_Mie_array=np.zeros((0, N_wl)),
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
    np.testing.assert_allclose(a, b, atol=1e-22, rtol=1e-11)


# ---- spectral contribution ---------------------------------------------------


@pytest.mark.parametrize("enable_haze,enable_deck", [(0, 0), (1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O"},
        {"contribution_species": "CH4"},
        {"contribution_species": "H2", "bulk_species": True},
        {"cloud_contribution": True},
    ],
    ids=["H2O", "CH4", "bulk", "cloud"],
)
def test_spectral_contribution_matches_poseidon(enable_haze, enable_deck, opts):
    cfg = _setup()
    cfg["enable_haze"] = enable_haze
    cfg["enable_deck"] = enable_deck

    ours = extinction_spectral_contribution(**{k: v for k, v in cfg.items()}, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


def test_spectral_contribution_He_branch():
    cfg = _setup()
    cfg["enable_haze"] = 1
    cfg["enable_deck"] = 1
    opts = {"contribution_species": "He"}
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- pressure contribution ---------------------------------------------------


@pytest.mark.parametrize("enable_haze,enable_deck", [(0, 0), (1, 1)])
@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O", "layer_to_ignore": 5},
        {"contribution_species": "CH4", "layer_to_ignore": 10},
        {
            "contribution_species": "H2",
            "bulk_species": True,
            "layer_to_ignore": 7,
        },
        {"cloud_contribution": True, "layer_to_ignore": 8},
        {"total_pressure_contribution": True, "layer_to_ignore": 6},
    ],
    ids=["H2O", "CH4", "bulk", "cloud", "total"],
)
def test_pressure_contribution_matches_poseidon(enable_haze, enable_deck, opts):
    cfg = _setup()
    cfg["enable_haze"] = enable_haze
    cfg["enable_deck"] = enable_deck

    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- gating tests ------------------------------------------------------------


def test_spectral_rejects_surface():
    cfg = _setup()
    cfg["enable_surface"] = 1
    with pytest.raises(NotImplementedError, match="surface"):
        extinction_spectral_contribution(**cfg, contribution_species="H2O")


def test_spectral_rejects_Mie():
    cfg = _setup()
    cfg["enable_Mie"] = 1
    with pytest.raises(NotImplementedError, match="Mie"):
        extinction_spectral_contribution(**cfg, contribution_species="H2O")


def test_pressure_rejects_surface():
    cfg = _setup()
    cfg["enable_surface"] = 1
    with pytest.raises(NotImplementedError, match="surface"):
        extinction_pressure_contribution(**cfg, total_pressure_contribution=True)


def test_pressure_rejects_Mie():
    cfg = _setup()
    cfg["enable_Mie"] = 1
    with pytest.raises(NotImplementedError, match="Mie"):
        extinction_pressure_contribution(**cfg, total_pressure_contribution=True)


# ---- compute_spectrum wiring -------------------------------------------------


def test_compute_spectrum_accepts_kappa_contributions(monkeypatch):
    """kappa_contributions bypasses extinction(...) wiring (POSEIDON
    core.py:1519-1521 behaviour).

    Passes a synthetic 4-tuple and monkeypatches `extinction` to raise;
    the test verifies the bypass branch fires (so `extinction` is never
    called) and that the call proceeds into the downstream TRIDENT
    setup (where it fails on the empty `planet`/`atmosphere` bundles —
    that crash is intentional, by `CLAUDE.md` §2 "let it crash").
    """
    from jaxposeidon import _compute_spectrum as cs

    called = {"extinction": False}

    def fake_extinction(*args, **kwargs):
        called["extinction"] = True
        raise AssertionError("extinction(...) should be bypassed")

    monkeypatch.setattr(cs, "extinction", fake_extinction)

    kg = np.zeros((2, 1, 1, 2))
    with pytest.raises(KeyError):
        cs.compute_spectrum(
            planet={},
            star={},
            model={
                "disable_atmosphere": False,
                "cloud_model": "cloud-free",
                "cloud_type": "",
                "chemical_species": [],
                "active_species": [],
                "CIA_pairs": [],
                "ff_pairs": [],
                "bf_species": [],
            },
            atmosphere={"is_physical": True},
            opac={
                "opacity_treatment": "opacity_sampling",
                "T_fine": np.array([500.0, 1000.0]),
            },
            wl=np.array([1.0, 2.0]),
            kappa_contributions=(kg, kg, kg, kg),
        )
    assert not called["extinction"]


# ---- H-minus ff/bf parity ----------------------------------------------------


def _setup_hminus(N_layers=14, N_wl=10, N_T_fine=5, N_P_fine=6, seed=1):
    """H-minus fixture: H- is in bf_species ONLY, not in chemical_species.

    This is POSEIDON's convention (H- is a photo-dissociation source,
    not a mixing-ratio species in `chemical_species`). It exercises the
    documented `MISMATCHES.md` numba-default-zero behaviour: when the
    contribution lookup loops at `contributions.py:215-222` /
    `:1207-1214` never find `contribution_species='H-'` in
    `chemical_species` or `active_species`, the kernel falls through
    on `contribution_molecule_*_index = 0` — matched here by the
    port's explicit `= 0` initialisation.
    """
    rng = np.random.default_rng(seed)
    chemical_species = np.array(["H2", "He", "H2O"])
    active_species = np.array(["H2O"])
    cia_pairs = np.array(["H2-H2", "H2-He"])
    ff_pairs = np.array(["H-ff"])
    bf_species = np.array(["H-bf"])
    aerosol_species = np.array([], dtype=str)

    P = np.logspace(np.log10(100.0), np.log10(1.0e-6), N_layers)
    T = 1500.0 + 100.0 * rng.standard_normal((N_layers, 1, 1))
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))
    wl = np.linspace(1.0, 5.0, N_wl)

    X = np.zeros((3, N_layers, 1, 1))
    X[0] = 0.84
    X[1] = 0.149
    X[2] = 0.005

    X_active = X[2:3]

    X_cia = np.zeros((2, 2, N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]

    n_H = rng.uniform(1e-7, 1e-5, size=(N_layers, 1, 1))
    n_e = rng.uniform(1e-12, 1e-10, size=(N_layers, 1, 1))
    n_Hm = rng.uniform(1e-10, 1e-8, size=(N_layers, 1, 1))
    X_ff = np.zeros((2, 1, N_layers, 1, 1))
    X_ff[0, 0] = n_H
    X_ff[1, 0] = n_e

    X_bf = np.zeros((1, N_layers, 1, 1))
    X_bf[0] = n_Hm

    T_fine = np.linspace(500.0, 2500.0, N_T_fine)
    log_P_fine = np.linspace(-6.0, 2.0, N_P_fine)

    sigma_stored = rng.uniform(0.0, 1e-22, size=(1, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(0.0, 1e-44, size=(2, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(3, N_wl))
    ff_stored = rng.uniform(0.0, 1e-44, size=(1, N_T_fine, N_wl))
    bf_stored = rng.uniform(0.0, 1e-20, size=(1, N_wl))

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
        P_surf=1.0e-30,
        enable_Mie=0,
        n_aerosol_array=np.zeros((0, N_layers, 1, 1)),
        sigma_Mie_array=np.zeros((0, N_wl)),
        P_deep=1000.0,
    )


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H-"},
        {"contribution_species": "H2O"},
        {"contribution_species": "H2", "bulk_species": True},
    ],
    ids=["H-", "H2O-with-ff-bf", "bulk-with-ff-bf"],
)
def test_spectral_contribution_ff_bf_matches_poseidon(opts):
    cfg = _setup_hminus()
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H-", "layer_to_ignore": 6},
        {"contribution_species": "H2O", "layer_to_ignore": 4},
        {"total_pressure_contribution": True, "layer_to_ignore": 8},
    ],
    ids=["H-", "H2O-with-ff-bf", "total-with-ff-bf"],
)
def test_pressure_contribution_ff_bf_matches_poseidon(opts):
    cfg = _setup_hminus()
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


# ---- multi-sector / multi-zone parity ---------------------------------------


def _setup_multi(N_sectors=2, N_zones=3, **kw):
    cfg = _setup(**kw)
    N_layers = cfg["n"].shape[0]
    cfg["N_sectors"] = N_sectors
    cfg["N_zones"] = N_zones
    rng = np.random.default_rng(42)
    cfg["T"] = 1000.0 + 200.0 * rng.standard_normal((N_layers, N_sectors, N_zones))
    cfg["n"] = rng.uniform(1e15, 1e25, size=(N_layers, N_sectors, N_zones))
    X = np.zeros((4, N_layers, N_sectors, N_zones))
    X[0] = 0.84
    X[1] = 0.149
    X[2] = 0.005
    X[3] = 0.006
    cfg["X"] = X
    cfg["X_active"] = X[2:4]
    X_cia = np.zeros((2, 2, N_layers, N_sectors, N_zones))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    cfg["X_cia"] = X_cia
    cfg["X_ff"] = np.zeros((2, 0, N_layers, N_sectors, N_zones))
    cfg["X_bf"] = np.zeros((0, N_layers, N_sectors, N_zones))
    return cfg


@pytest.mark.parametrize(
    "opts",
    [
        {"contribution_species": "H2O"},
        {"contribution_species": "H2", "bulk_species": True},
        {"cloud_contribution": True},
    ],
    ids=["H2O", "bulk", "cloud"],
)
def test_spectral_contribution_multi_sector_zone(opts):
    cfg = _setup_multi()
    cfg["enable_haze"] = 1
    cfg["enable_deck"] = 1
    ours = extinction_spectral_contribution(**cfg, **opts)
    theirs = _call_poseidon_spec(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)


def test_pressure_contribution_multi_sector_zone():
    cfg = _setup_multi()
    cfg["enable_haze"] = 1
    cfg["enable_deck"] = 1
    opts = {"total_pressure_contribution": True, "layer_to_ignore": 7}
    ours = extinction_pressure_contribution(**cfg, **opts)
    theirs = _call_poseidon_press(cfg, **opts)
    for a, b in zip(ours, theirs):
        _assert_close(a, b)
