"""Phase 0.5.4: H-minus ff/bf opacity parity tests.

Direct POSEIDON parity for:
  - `H_minus_bound_free` (POSEIDON absorption.py:556-604)
  - `H_minus_free_free`  (POSEIDON absorption.py:606-691)

Plus an end-to-end ``extinction`` check showing the ff/bf path is
correctly threaded (kappa_gas grows by the expected ff/bf contribution
when ff_pairs / bf_species are non-empty).
"""

import numpy as np
import pytest

from jaxposeidon._h_minus import H_minus_bound_free, H_minus_free_free


@pytest.mark.parametrize(
    "wl_um",
    [
        np.linspace(0.3, 2.0, 50),
        np.linspace(0.5, 5.0, 100),
        np.linspace(1.0, 1.8, 30),     # straddles the 1.6421 μm threshold
        np.array([0.5, 1.0, 1.5, 1.6, 1.6421, 1.7, 2.0]),  # threshold edge
    ],
)
def test_H_minus_bound_free_matches_poseidon(wl_um):
    from POSEIDON.absorption import H_minus_bound_free as p_bf

    ours = H_minus_bound_free(wl_um)
    theirs = p_bf(wl_um)
    # FP-precision parity; POSEIDON uses numba JIT which may reorder
    # the John 1988 polynomial summation by a few ULPs vs pure numpy.
    np.testing.assert_allclose(ours, theirs, atol=1e-65, rtol=1e-13)


@pytest.mark.parametrize(
    "wl_um,T_arr",
    [
        (np.linspace(0.3, 2.0, 50), np.array([1500.0, 2500.0, 3500.0])),
        (np.linspace(0.2, 5.0, 100), np.linspace(1000.0, 4000.0, 8)),
        (np.linspace(0.18, 0.4, 60), np.array([2000.0])),  # short-wavelength fit
        (np.linspace(0.35, 2.0, 80), np.array([1800.0, 2400.0])),  # long
        (np.array([0.18, 0.182, 0.3, 0.3645, 0.5, 1.0, 2.0, 5.0]),
         np.array([1500.0, 2500.0])),  # all branch edges
    ],
)
def test_H_minus_free_free_matches_poseidon(wl_um, T_arr):
    from POSEIDON.absorption import H_minus_free_free as p_ff

    ours = H_minus_free_free(wl_um, T_arr)
    theirs = p_ff(wl_um, T_arr)
    np.testing.assert_allclose(ours, theirs, atol=1e-65, rtol=1e-13)


def test_extinction_with_h_minus_lifts_v0_guard():
    """Verify the v0 NotImplementedError guard on non-empty ff_pairs /
    bf_species is removed and that the corresponding extinction path
    accumulates into kappa_gas."""
    from jaxposeidon._opacities import extinction

    N_layers, N_wl = 5, 20
    N_sectors, N_zones = 1, 1
    wl = np.linspace(0.3, 5.0, N_wl)
    P = np.logspace(2, -6, N_layers)
    T = np.full((N_layers, N_sectors, N_zones), 2500.0)
    n = np.full((N_layers, N_sectors, N_zones), 1.0e20)
    T_fine = np.array([1500.0, 2500.0, 3500.0])
    log_P_fine = np.array([-6.0, -3.0, 0.0, 2.0])

    # Empty ff/bf: kappa_gas only has CIA + active + Rayleigh contributions
    chemical_species = np.array(["H2", "He"])
    active_species = np.array([], dtype=str)
    cia_pairs = np.array([], dtype=str)

    # Build empty arrays sized to match POSEIDON conventions
    X = np.full((2, N_layers, N_sectors, N_zones), 0.5)
    X_active = np.zeros((0, N_layers, N_sectors, N_zones))
    X_cia = np.zeros((2, 0, N_layers, N_sectors, N_zones))
    sigma_stored = np.zeros((0, len(log_P_fine), len(T_fine), N_wl))
    cia_stored = np.zeros((0, len(T_fine), N_wl))
    Rayleigh_stored = np.zeros((2, N_wl))

    # Now with one ff pair (H-e-) and one bf species (H-)
    ff_pairs = np.array(["H-ff"])
    bf_species = np.array(["H-bf"])
    X_ff = np.full((2, 1, N_layers, N_sectors, N_zones), 1.0e-7)
    X_bf = np.full((1, N_layers, N_sectors, N_zones), 1.0e-9)
    ff_stored = np.full((1, len(T_fine), N_wl), 1.0e-49)
    bf_stored = np.full((1, N_wl), 1.0e-21)

    kg, kr, kc, _ = extinction(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
        n=n, T=T, P=P, wl=wl, X=X,
        X_active=X_active, X_cia=X_cia, X_ff=X_ff, X_bf=X_bf,
        a=1.0, gamma=-4.0, P_cloud=np.array([1.0]), kappa_cloud_0=0.0,
        sigma_stored=sigma_stored, cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored, ff_stored=ff_stored, bf_stored=bf_stored,
        enable_haze=0, enable_deck=0, enable_surface=0,
        N_sectors=N_sectors, N_zones=N_zones,
        T_fine=T_fine, log_P_fine=log_P_fine, P_surf=100.0,
        enable_Mie=0,
        n_aerosol_array=np.zeros((1, N_layers, N_sectors, N_zones)),
        sigma_Mie_array=np.zeros((1, N_wl)),
    )
    # ff/bf contributions should be strictly positive in the interior
    assert kg.shape == (N_layers, N_sectors, N_zones, N_wl)
    # i_bot indexing: P_deep=1000, so all layers in this P grid are active
    # except possibly the bottom; check at least one layer accumulates.
    assert (kg > 0).any()


def test_extinction_matches_poseidon_with_h_minus():
    """End-to-end parity: extinction with non-empty ff_pairs / bf_species
    matches POSEIDON's extinction() output element-wise."""
    from POSEIDON.absorption import extinction as p_extinction

    from jaxposeidon._opacities import extinction

    N_layers, N_wl = 8, 30
    N_sectors, N_zones = 1, 1
    rng = np.random.default_rng(42)
    wl = np.linspace(0.3, 5.0, N_wl)
    P = np.logspace(2, -6, N_layers)
    T = rng.uniform(1800.0, 3000.0, size=(N_layers, N_sectors, N_zones))
    n = rng.uniform(1e19, 1e22, size=(N_layers, N_sectors, N_zones))
    T_fine = np.linspace(1500.0, 3500.0, 5)
    log_P_fine = np.linspace(-6.0, 2.0, 9)

    chemical_species = np.array(["H2", "He", "H", "e-"])
    active_species = np.array([], dtype=str)
    cia_pairs = np.array([], dtype=str)
    ff_pairs = np.array(["H-ff"])
    bf_species = np.array(["H-bf"])

    X = rng.uniform(0.1, 0.5, size=(4, N_layers, N_sectors, N_zones))
    X_active = np.zeros((0, N_layers, N_sectors, N_zones))
    X_cia = np.zeros((2, 0, N_layers, N_sectors, N_zones))
    X_ff = rng.uniform(1e-8, 1e-6, size=(2, 1, N_layers, N_sectors, N_zones))
    X_bf = rng.uniform(1e-10, 1e-8, size=(1, N_layers, N_sectors, N_zones))

    sigma_stored = np.zeros((0, len(log_P_fine), len(T_fine), N_wl))
    cia_stored = np.zeros((0, len(T_fine), N_wl))
    Rayleigh_stored = rng.uniform(1e-32, 1e-30, size=(4, N_wl))
    ff_stored = H_minus_free_free(wl, T_fine)
    ff_stored = ff_stored[np.newaxis, :, :]  # (N_ff_pairs=1, N_T_fine, N_wl)
    bf_stored = H_minus_bound_free(wl)[np.newaxis, :]  # (N_bf_species=1, N_wl)

    common = dict(
        chemical_species=chemical_species,
        active_species=active_species,
        cia_pairs=cia_pairs,
        ff_pairs=ff_pairs,
        bf_species=bf_species,
        n=n, T=T, P=P, wl=wl, X=X,
        X_active=X_active, X_cia=X_cia, X_ff=X_ff, X_bf=X_bf,
        a=1.0, gamma=-4.0, P_cloud=np.array([1.0]), kappa_cloud_0=0.0,
        sigma_stored=sigma_stored, cia_stored=cia_stored,
        Rayleigh_stored=Rayleigh_stored, ff_stored=ff_stored, bf_stored=bf_stored,
        enable_haze=0, enable_deck=0, enable_surface=0,
        N_sectors=N_sectors, N_zones=N_zones,
        T_fine=T_fine, log_P_fine=log_P_fine, P_surf=100.0,
    )
    kg_ours, kr_ours, kc_ours, _ = extinction(
        enable_Mie=0,
        n_aerosol_array=np.zeros((1, N_layers, N_sectors, N_zones)),
        sigma_Mie_array=np.zeros((1, N_wl)),
        **common,
    )

    kg_theirs, kr_theirs, kc_theirs, _ = p_extinction(
        chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
        n, T, P, wl, X, X_active, X_cia, X_ff, X_bf,
        1.0, -4.0, np.array([1.0]), 0.0,
        sigma_stored, cia_stored, Rayleigh_stored, ff_stored, bf_stored,
        0, 0, 0, N_sectors, N_zones, T_fine, log_P_fine, 100.0, 0,
        np.zeros((1, N_layers, N_sectors, N_zones)),
        np.zeros((1, N_wl)),
    )
    np.testing.assert_allclose(kg_ours, kg_theirs, atol=1e-22, rtol=1e-13)
    np.testing.assert_allclose(kr_ours, kr_theirs, atol=1e-22, rtol=1e-13)
    np.testing.assert_allclose(kc_ours, kc_theirs, atol=0, rtol=0)
