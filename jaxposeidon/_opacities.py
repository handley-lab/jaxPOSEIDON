"""Runtime extinction — assemble κ from pre-interpolated cross sections.

Faithful port of the v0 subset of POSEIDON `absorption.py:1034-1227`
(`extinction(...)`). Mirrors POSEIDON's nearest-fine-grid lookup in
(log_P, T) and the MacMad17 deck + haze additions.

v0 envelope:
- enable_haze in {0, 1}
- enable_deck in {0, 1}
- enable_surface = 0 (deferred)
- enable_Mie = 0 (deferred)
- N_ff_pairs = 0 (H-minus deferred)
- N_bf_species = 0 (H-minus deferred)
- ff_stored, bf_stored may be supplied but ignored
- N_sectors = N_zones = 1 (1D atmosphere)

CIA and active-species opacity are the heaviest branches and are ported
in full. The output 4-tuple matches POSEIDON's:
    (kappa_gas, kappa_Ray, kappa_cloud, kappa_cloud_separate)
"""

import numpy as np

from jaxposeidon._opacity_precompute import closest_index


def extinction(chemical_species, active_species, cia_pairs, ff_pairs, bf_species,
               n, T, P, wl, X, X_active, X_cia, X_ff, X_bf,
               a, gamma, P_cloud, kappa_cloud_0,
               sigma_stored, cia_stored, Rayleigh_stored, ff_stored, bf_stored,
               enable_haze, enable_deck, enable_surface,
               N_sectors, N_zones, T_fine, log_P_fine, P_surf,
               enable_Mie, n_aerosol_array, sigma_Mie_array, P_deep=1000.0):
    """Compute kappa_gas, kappa_Ray, kappa_cloud arrays.

    Bit-exact port of POSEIDON `absorption.py:1034-1227` for the v0
    envelope (no surface, no Mie). Surface, Mie, and ff/bf-active
    configurations raise NotImplementedError.
    """
    if enable_surface == 1:
        raise NotImplementedError("surfaces deferred to v1")
    if enable_Mie == 1:
        raise NotImplementedError("Mie clouds deferred to v1")
    if len(ff_pairs) > 0 or len(bf_species) > 0:
        raise NotImplementedError("H-minus ff/bf opacity deferred to v1")

    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_wl = len(wl)
    N_layers = len(P)

    kappa_gas = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud_separate = np.zeros(
        (len(n_aerosol_array), N_layers, N_sectors, N_zones, N_wl)
    )

    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)

    i_bot = int(np.argmin(np.abs(P - P_deep)))

    if enable_haze == 1:
        slope = np.power(wl / 0.35, gamma)

    for j in range(N_sectors):
        for k in range(N_zones):
            for i in range(i_bot, N_layers):
                n_level = n[i, j, k]
                idx_T_fine = closest_index(T[i, j, k], T_fine[0], T_fine[-1], N_T_fine)
                idx_P_fine = closest_index(np.log10(P[i]),
                                            log_P_fine[0], log_P_fine[-1],
                                            N_P_fine)

                for q in range(N_cia_pairs):
                    n_cia_1 = n_level * X_cia[0, q, i, j, k]
                    n_cia_2 = n_level * X_cia[1, q, i, j, k]
                    n_n_cia = n_cia_1 * n_cia_2
                    kappa_gas[i, j, k, :] += n_n_cia * cia_stored[q, idx_T_fine, :]

                for q in range(N_species_active):
                    n_q = n_level * X_active[q, i, j, k]
                    kappa_gas[i, j, k, :] += n_q * sigma_stored[q, idx_P_fine, idx_T_fine, :]

                for q in range(N_species):
                    n_q = n_level * X[q, i, j, k]
                    kappa_Ray[i, j, k, :] += n_q * Rayleigh_stored[q, :]

            if enable_haze == 1:
                for i in range(i_bot, N_layers):
                    haze_amp = n[i, j, k] * a * 5.31e-31
                    kappa_cloud[i, j, k, :] += haze_amp * slope

            if enable_deck == 1:
                kappa_cloud[P > P_cloud[0], j, k, :] += kappa_cloud_0

    return kappa_gas, kappa_Ray, kappa_cloud, kappa_cloud_separate
