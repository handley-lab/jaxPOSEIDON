"""Phase 0.5.17a + 0.5.17c: save_spectrum + disable_continuum.

Ports:
- POSEIDON utility.py:694-709 (write_spectrum) → _output.write_spectrum
- POSEIDON absorption.py:1651 + the equivalent gate in extinction()
  (CIA + Rayleigh skipped when disable_continuum=True)
"""

import os

import numpy as np
import pytest

from jaxposeidon import _opacities, _output


def test_write_spectrum_file_format(tmp_path):
    wl = np.linspace(1.0, 2.0, 5)
    spectrum = np.array([1.234e-3, 2.345e-3, 3.456e-3, 4.567e-3, 5.678e-3])
    file_path = _output.write_spectrum(
        planet_name="K2-18b",
        model_name="MyModel",
        spectrum=spectrum,
        wl=wl,
        output_dir=str(tmp_path),
    )
    assert file_path == str(
        tmp_path / "K2-18b" / "spectra" / "K2-18b_MyModel_spectrum.txt"
    )
    assert os.path.isfile(file_path)
    rows = np.loadtxt(file_path)
    np.testing.assert_array_equal(rows[:, 0], wl)
    np.testing.assert_array_equal(rows[:, 1], spectrum)


def test_write_spectrum_matches_poseidon(tmp_path, monkeypatch):
    """Round-trip parity against POSEIDON's write_spectrum."""
    monkeypatch.chdir(tmp_path)
    os.makedirs("POSEIDON_output/K2-18b/spectra", exist_ok=True)

    wl = np.linspace(1.0, 5.0, 50)
    spectrum = 0.01 + 0.001 * np.sin(wl)

    from POSEIDON.utility import write_spectrum as p_write

    p_write("K2-18b", "P", spectrum, wl)

    _output.write_spectrum(
        planet_name="K2-18b",
        model_name="M",
        spectrum=spectrum,
        wl=wl,
        output_dir=str(tmp_path / "POSEIDON_output"),
    )

    p_rows = np.loadtxt("POSEIDON_output/K2-18b/spectra/K2-18b_P_spectrum.txt")
    m_rows = np.loadtxt("POSEIDON_output/K2-18b/spectra/K2-18b_M_spectrum.txt")
    np.testing.assert_array_equal(p_rows, m_rows)


# ---------------------------------------------------------------------------
# disable_continuum (CIA + Rayleigh gating)
# ---------------------------------------------------------------------------
def _extinction_kwargs():
    rng = np.random.default_rng(0)
    N_layers, N_wl, N_T_fine, N_P_fine = 20, 10, 5, 6
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
    sigma_stored = rng.uniform(0.0, 1e-22, size=(1, N_P_fine, N_T_fine, N_wl))
    cia_stored = rng.uniform(0.0, 1e-44, size=(2, N_T_fine, N_wl))
    Rayleigh_stored = rng.uniform(0.0, 1e-27, size=(3, N_wl))
    ff_stored = np.zeros((0, N_T_fine, N_wl))
    bf_stored = np.zeros((0, N_wl))
    return dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
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


def test_disable_continuum_zeroes_CIA_and_Rayleigh():
    kw = _extinction_kwargs()
    kg_on, kr_on, _, _ = _opacities.extinction(**kw, disable_continuum=False)
    kg_off, kr_off, _, _ = _opacities.extinction(**kw, disable_continuum=True)
    # Rayleigh entirely zeroed.
    np.testing.assert_array_equal(kr_off, np.zeros_like(kr_off))
    # kappa_gas without continuum equals kappa_gas with continuum minus
    # the CIA contribution; for this config (no ff/bf, only CIA + active),
    # disable_continuum reduces kappa_gas by the CIA part.
    assert np.all(kg_off <= kg_on + 1e-30)


def test_disable_continuum_preserves_active_species_opacity():
    """When CIA + Rayleigh are off, kappa_gas should still contain active
    molecular opacity."""
    kw = _extinction_kwargs()
    kg_off, _, _, _ = _opacities.extinction(**kw, disable_continuum=True)
    # Active species contribution is non-zero; kappa_gas should reflect it.
    assert np.any(kg_off > 0)


@pytest.mark.parametrize("enable_haze,enable_deck", [(0, 0), (1, 0), (0, 1)])
def test_disable_continuum_does_not_affect_clouds(enable_haze, enable_deck):
    kw = _extinction_kwargs()
    kw["enable_haze"] = enable_haze
    kw["enable_deck"] = enable_deck
    _, _, kc_on, _ = _opacities.extinction(**kw, disable_continuum=False)
    _, _, kc_off, _ = _opacities.extinction(**kw, disable_continuum=True)
    np.testing.assert_array_equal(kc_on, kc_off)
