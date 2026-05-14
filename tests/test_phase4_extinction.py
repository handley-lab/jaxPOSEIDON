"""Phase 4 extinction tests against POSEIDON `absorption.extinction`."""

import numpy as np
import pytest

from jaxposeidon._opacities import extinction


def _set_up_extinction(enable_haze=0, enable_deck=0,
                       N_layers=20, N_wl=10, N_T_fine=5, N_P_fine=6):
    rng = np.random.default_rng(0)
    chemical_species = np.array(["H2", "He", "H2O"])
    active_species = np.array(["H2O"])
    cia_pairs = np.array(["H2-H2", "H2-He"])
    ff_pairs = np.array([], dtype=str)
    bf_species = np.array([], dtype=str)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-6), N_layers)
    T = 1000.0 * np.ones((N_layers, 1, 1))
    n = rng.uniform(1e15, 1e25, size=(N_layers, 1, 1))
    wl = np.linspace(1.0, 5.0, N_wl)
    X = np.zeros((3, N_layers, 1, 1))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 0.001
    X_active = X[2:3]
    X_cia = np.zeros((2, 2, N_layers, 1, 1))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]
    X_ff = np.zeros((2, 0, N_layers, 1, 1))
    X_bf = np.zeros((0, N_layers, 1, 1))
    T_fine = np.linspace(500.0, 2000.0, N_T_fine)
    log_P_fine = np.linspace(-6.0, 2.0, N_P_fine)
    sigma_stored = rng.uniform(0.0, 1e-22,
                                size=(1, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(0.0, 1e-44, size=(2, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(3, N_wl))
    ff_stored = np.zeros((0, N_T_fine, N_wl))
    bf_stored = np.zeros((0, N_wl))
    return dict(
        chemical_species=chemical_species, active_species=active_species,
        cia_pairs=cia_pairs, ff_pairs=ff_pairs, bf_species=bf_species,
        n=n, T=T, P=P, wl=wl, X=X, X_active=X_active, X_cia=X_cia,
        X_ff=X_ff, X_bf=X_bf,
        a=1.0, gamma=-4.0, P_cloud=np.array([1.0e-3]), kappa_cloud_0=1.0e-30,
        sigma_stored=sigma_stored, cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored, ff_stored=ff_stored, bf_stored=bf_stored,
        enable_haze=enable_haze, enable_deck=enable_deck, enable_surface=0,
        N_sectors=1, N_zones=1, T_fine=T_fine, log_P_fine=log_P_fine,
        P_surf=1.0e-30,
        enable_Mie=0, n_aerosol_array=np.zeros((0, N_layers, 1, 1)),
        sigma_Mie_array=np.zeros((0, N_wl)), P_deep=1000.0,
    )


@pytest.mark.parametrize("enable_haze,enable_deck", [
    (0, 0), (1, 0), (0, 1), (1, 1),
])
def test_extinction_matches_poseidon(enable_haze, enable_deck):
    from POSEIDON.absorption import extinction as p_ext
    cfg = _set_up_extinction(enable_haze=enable_haze, enable_deck=enable_deck)
    ours = extinction(**cfg)
    theirs = p_ext(
        cfg["chemical_species"], cfg["active_species"],
        cfg["cia_pairs"], cfg["ff_pairs"], cfg["bf_species"],
        cfg["n"], cfg["T"], cfg["P"], cfg["wl"], cfg["X"],
        cfg["X_active"], cfg["X_cia"], cfg["X_ff"], cfg["X_bf"],
        cfg["a"], cfg["gamma"], cfg["P_cloud"], cfg["kappa_cloud_0"],
        cfg["sigma_stored"], cfg["cia_stored"], cfg["Rayleigh_stored"],
        cfg["ff_stored"], cfg["bf_stored"],
        cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
        cfg["N_sectors"], cfg["N_zones"], cfg["T_fine"], cfg["log_P_fine"],
        cfg["P_surf"], cfg["enable_Mie"], cfg["n_aerosol_array"],
        cfg["sigma_Mie_array"], cfg["P_deep"],
    )
    # Compare all four outputs at FP precision. POSEIDON's numba path may
    # reduce in a slightly different order than numpy/JAX, producing
    # ULP-scale residuals (~4e-25 absolute) on the (1,1) parametric case.
    for a, b in zip(ours, theirs):
        np.testing.assert_allclose(a, b, atol=1e-22, rtol=1e-13)


def test_extinction_rejects_surface():
    cfg = _set_up_extinction()
    cfg["enable_surface"] = 1
    with pytest.raises(NotImplementedError, match="surface"):
        extinction(**cfg)


def test_extinction_rejects_Mie():
    cfg = _set_up_extinction()
    cfg["enable_Mie"] = 1
    with pytest.raises(NotImplementedError, match="Mie"):
        extinction(**cfg)


# Phase 0.5.4: H-minus ff/bf are now ported (see _h_minus.py +
# tests/test_phase_v05_4_h_minus.py); the v0 rejection tests for
# `ff_pairs` / `bf_species` non-empty are obsolete.
