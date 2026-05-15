"""Spectral and pressure contribution kernels.

Faithful numpy port of POSEIDON `contributions.py`:

- `extinction_spectral_contribution` mirrors POSEIDON
  ``contributions.extinction_spectral_contribution`` (the kappa-generator
  used by ``spectral_contribution`` to isolate one species' opacity).
- `extinction_pressure_contribution` mirrors POSEIDON
  ``contributions.extinction_pressure_contribution`` (the kappa-generator
  used by ``pressure_contribution_compute_spectrum`` to suppress one
  layer's contribution).

All CIA / free-free / bound-free / active / Rayleigh / haze / deck /
surface / Mie branches are ported verbatim from POSEIDON. The
surface branch sets ``kappa_gas`` to ``1e250`` below ``P_surf``;
the Mie branch handles both the opaque-deck-first-element layout
(``len(n_aerosol_array) == len(sigma_Mie_array) + 1``) and the
no-opaque-deck layout (lengths equal), and supports
``cloud_contribution`` / ``cloud_total_contribution`` selectors
with the ``aerosol_species_index`` lookup against
``aerosol_species``.

The orchestrators (``spectral_contribution`` /
``pressure_contribution`` / ``pressure_contribution_compute_spectrum``)
re-invoke ``compute_spectrum`` once per requested species and are
integration glue — they are deferred to a later phase and not ported
here.
"""

import numpy as np

from jaxposeidon._opacity_precompute import closest_index


def _bulk_metadata(chemical_species, active_species, cia_pairs):
    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_bulk_species = N_species - N_species_active
    bulk_species_indices = range(N_bulk_species)
    bulk_species_names = chemical_species[:N_bulk_species]

    bulk_cia_indices = []
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split("-")
        pair_1_bool = False
        pair_2_bool = False
        for j in bulk_species_names:
            if pair_1 == j:
                pair_1_bool = True
            if pair_2 == j:
                pair_2_bool = True
        if pair_1_bool and pair_2_bool:
            bulk_cia_indices.append(i)

    return N_bulk_species, bulk_species_indices, bulk_cia_indices


def _cia_indices_for(cia_pairs, contribution_species):
    cia_indices = []
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split("-")
        if contribution_species == pair_1 or contribution_species == pair_2:
            cia_indices.append(i)
    return cia_indices


def extinction_spectral_contribution(
    chemical_species,
    active_species,
    cia_pairs,
    ff_pairs,
    bf_species,
    aerosol_species,
    n,
    T,
    P,
    wl,
    X,
    X_active,
    X_cia,
    X_ff,
    X_bf,
    a,
    gamma,
    P_cloud,
    kappa_cloud_0,
    sigma_stored,
    cia_stored,
    Rayleigh_stored,
    ff_stored,
    bf_stored,
    enable_haze,
    enable_deck,
    enable_surface,
    N_sectors,
    N_zones,
    T_fine,
    log_P_fine,
    P_surf,
    enable_Mie,
    n_aerosol_array,
    sigma_Mie_array,
    P_deep=1000.0,
    contribution_species="",
    bulk_species=False,
    cloud_contribution=False,
    cloud_species="",
    cloud_total_contribution=False,
):
    """Per-species kappa generator for spectral contribution plots.

    Mirrors POSEIDON ``contributions.py:143-505``
    (``extinction_spectral_contribution``) including surface and Mie
    branches.
    """
    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)

    _, bulk_species_indices, bulk_cia_indices = _bulk_metadata(
        chemical_species, active_species, cia_pairs
    )

    contribution_molecule_species_index = 0
    contribution_molecule_active_index = 0
    cia_indices = []
    if not bulk_species and not cloud_contribution:
        for i in range(len(chemical_species)):
            if contribution_species == chemical_species[i]:
                contribution_molecule_species_index = i

        for i in range(len(active_species)):
            if contribution_species == active_species[i]:
                contribution_molecule_active_index = i

        cia_indices = _cia_indices_for(cia_pairs, contribution_species)

    bound_free = contribution_species == "H-"

    N_wl = len(wl)
    N_layers = len(P)

    kappa_gas = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros((N_layers, N_sectors, N_zones, N_wl))

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
                idx_P_fine = closest_index(
                    np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine
                )

                for q in range(N_cia_pairs):
                    if not bulk_species and not cloud_contribution:
                        if q in bulk_cia_indices or q in cia_indices:
                            n_cia_1 = n_level * X_cia[0, q, i, j, k]
                            n_cia_2 = n_level * X_cia[1, q, i, j, k]
                            n_n_cia = n_cia_1 * n_cia_2
                        else:
                            n_n_cia = 0.0
                    else:
                        if q in bulk_cia_indices:
                            n_cia_1 = n_level * X_cia[0, q, i, j, k]
                            n_cia_2 = n_level * X_cia[1, q, i, j, k]
                            n_n_cia = n_cia_1 * n_cia_2
                        else:
                            n_n_cia = 0.0

                    kappa_gas[i, j, k, :] += n_n_cia * cia_stored[q, idx_T_fine, :]

                for q in range(N_ff_pairs):
                    n_ff_1 = n_level * X_ff[0, q, i, j, k]
                    n_ff_2 = n_level * X_ff[1, q, i, j, k]
                    n_n_ff = n_ff_1 * n_ff_2
                    kappa_gas[i, j, k, :] += n_n_ff * ff_stored[q, idx_T_fine, :]

                for q in range(N_bf_species):
                    if bound_free:
                        n_q = n_level * X_bf[q, i, j, k]
                    else:
                        n_q = 0.0
                    kappa_gas[i, j, k, :] += n_q * bf_stored[q, :]

                for q in range(N_species_active):
                    if not bulk_species and not cloud_contribution and not bound_free:
                        if contribution_species == "He":
                            n_q = 0.0
                        elif q == contribution_molecule_active_index:
                            n_q = n_level * X_active[q, i, j, k]
                        else:
                            n_q = 0.0
                    else:
                        n_q = 0.0
                    kappa_gas[i, j, k, :] += (
                        n_q * sigma_stored[q, idx_P_fine, idx_T_fine, :]
                    )

                for q in range(N_species):
                    if not bulk_species and not cloud_contribution and not bound_free:
                        if q == contribution_molecule_species_index:
                            n_q = n_level * X[q, i, j, k]
                        elif q in bulk_species_indices:
                            n_q = n_level * X[q, i, j, k]
                        else:
                            n_q = 0.0
                    else:
                        if q in bulk_species_indices:
                            n_q = n_level * X[q, i, j, k]
                        else:
                            n_q = 0.0
                    kappa_Ray[i, j, k, :] += n_q * Rayleigh_stored[q, :]

            if enable_haze == 1:
                for i in range(i_bot, N_layers):
                    haze_amp = n[i, j, k] * a * 5.31e-31
                    for l in range(N_wl):
                        if not cloud_contribution:
                            haze_amp_l = 0.0
                        else:
                            haze_amp_l = haze_amp
                        kappa_cloud[i, j, k, l] += haze_amp_l * slope[l]

            if enable_deck == 1:
                if not cloud_contribution:
                    kappa_cloud_0_eff = 0.0
                else:
                    kappa_cloud_0_eff = kappa_cloud_0
                kappa_cloud[(P > P_cloud[0]), j, k, :] += kappa_cloud_0_eff

            if enable_surface == 1:
                kappa_gas[(P > P_surf), j, k, :] = 1.0e250

            if enable_Mie == 1:
                aerosol_species_index = 0
                if cloud_contribution and not cloud_total_contribution:
                    for q in range(len(aerosol_species)):
                        if cloud_species == aerosol_species[q]:
                            aerosol_species_index = q

                if len(n_aerosol_array) == len(sigma_Mie_array):
                    for aerosol in range(len(n_aerosol_array)):
                        for i in range(i_bot, N_layers):
                            for l in range(len(wl)):
                                if not cloud_contribution:
                                    kappa_cloud[i, j, k, l] += (
                                        n_aerosol_array[aerosol][i, j, k] * 0.0
                                    )
                                elif (
                                    cloud_contribution and not cloud_total_contribution
                                ):
                                    if aerosol == aerosol_species_index:
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k]
                                            * sigma_Mie_array[aerosol][l]
                                        )
                                    else:
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k] * 0.0
                                        )
                                elif cloud_total_contribution:
                                    kappa_cloud[i, j, k, l] += (
                                        n_aerosol_array[aerosol][i, j, k]
                                        * sigma_Mie_array[aerosol][l]
                                    )
                else:
                    for aerosol in range(len(n_aerosol_array)):
                        if not cloud_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += 0.0
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k] * 0.0
                                        )
                        elif cloud_contribution and not cloud_total_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += 0.0
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        if aerosol - 1 == aerosol_species_index:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k]
                                                * sigma_Mie_array[aerosol - 1][l]
                                            )
                                        else:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k] * 0.0
                                            )
                        elif cloud_total_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += 1.0e250
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k]
                                            * sigma_Mie_array[aerosol - 1][l]
                                        )

    return kappa_gas, kappa_Ray, kappa_cloud


def extinction_pressure_contribution(
    chemical_species,
    active_species,
    cia_pairs,
    ff_pairs,
    bf_species,
    aerosol_species,
    n,
    T,
    P,
    wl,
    X,
    X_active,
    X_cia,
    X_ff,
    X_bf,
    a,
    gamma,
    P_cloud,
    kappa_cloud_0,
    sigma_stored,
    cia_stored,
    Rayleigh_stored,
    ff_stored,
    bf_stored,
    enable_haze,
    enable_deck,
    enable_surface,
    N_sectors,
    N_zones,
    T_fine,
    log_P_fine,
    P_surf,
    enable_Mie,
    n_aerosol_array,
    sigma_Mie_array,
    P_deep=1000.0,
    contribution_species="",
    bulk_species=False,
    cloud_contribution=False,
    cloud_species="",
    cloud_total_contribution=False,
    layer_to_ignore=0,
    total_pressure_contribution=False,
):
    """Per-layer kappa generator for pressure contribution plots.

    Mirrors POSEIDON ``contributions.py:1135-1550``
    (``extinction_pressure_contribution``) including surface and Mie
    branches.
    """
    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)

    _, bulk_species_indices, bulk_cia_indices = _bulk_metadata(
        chemical_species, active_species, cia_pairs
    )

    contribution_molecule_species_index = 0
    contribution_molecule_active_index = 0
    cia_indices = []
    if not bulk_species and not cloud_contribution:
        for i in range(len(chemical_species)):
            if contribution_species == chemical_species[i]:
                contribution_molecule_species_index = i

        for i in range(len(active_species)):
            if contribution_species == active_species[i]:
                contribution_molecule_active_index = i

        cia_indices = _cia_indices_for(cia_pairs, contribution_species)

    bound_free = contribution_species == "H-"

    N_wl = len(wl)
    N_layers = len(P)

    kappa_gas = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros((N_layers, N_sectors, N_zones, N_wl))

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
                idx_P_fine = closest_index(
                    np.log10(P[i]), log_P_fine[0], log_P_fine[-1], N_P_fine
                )

                for q in range(N_cia_pairs):
                    if (
                        bulk_species
                        and (q in bulk_species_indices)
                        and (i == layer_to_ignore)
                    ):
                        n_n_cia = 0.0
                    elif (
                        (not bulk_species)
                        and (not cloud_contribution)
                        and (not total_pressure_contribution)
                    ):
                        if (q in cia_indices) and (i == layer_to_ignore):
                            n_n_cia = 0.0
                        else:
                            n_cia_1 = n_level * X_cia[0, q, i, j, k]
                            n_cia_2 = n_level * X_cia[1, q, i, j, k]
                            n_n_cia = n_cia_1 * n_cia_2
                    elif total_pressure_contribution and (i == layer_to_ignore):
                        n_n_cia = 0.0
                    else:
                        n_cia_1 = n_level * X_cia[0, q, i, j, k]
                        n_cia_2 = n_level * X_cia[1, q, i, j, k]
                        n_n_cia = n_cia_1 * n_cia_2

                    kappa_gas[i, j, k, :] += n_n_cia * cia_stored[q, idx_T_fine, :]

                for q in range(N_ff_pairs):
                    if total_pressure_contribution and (i == layer_to_ignore):
                        n_n_ff = 0.0
                    else:
                        n_ff_1 = n_level * X_ff[0, q, i, j, k]
                        n_ff_2 = n_level * X_ff[1, q, i, j, k]
                        n_n_ff = n_ff_1 * n_ff_2
                    kappa_gas[i, j, k, :] += n_n_ff * ff_stored[q, idx_T_fine, :]

                for q in range(N_bf_species):
                    if total_pressure_contribution and (i == layer_to_ignore):
                        n_q = 0.0
                    elif bound_free and (i == layer_to_ignore):
                        n_q = 0.0
                    else:
                        n_q = n_level * X_bf[q, i, j, k]
                    kappa_gas[i, j, k, :] += n_q * bf_stored[q, :]

                for q in range(N_species_active):
                    if (
                        (not bulk_species)
                        and (not cloud_contribution)
                        and (not total_pressure_contribution)
                    ):
                        if contribution_species == "He":
                            n_q = 0.0
                        elif (q == contribution_molecule_active_index) and (
                            i == layer_to_ignore
                        ):
                            n_q = 0.0
                        else:
                            n_q = n_level * X_active[q, i, j, k]
                    elif total_pressure_contribution and (i == layer_to_ignore):
                        n_q = 0.0
                    else:
                        n_q = n_level * X_active[q, i, j, k]
                    kappa_gas[i, j, k, :] += (
                        n_q * sigma_stored[q, idx_P_fine, idx_T_fine, :]
                    )

                for q in range(N_species):
                    if (
                        (not bulk_species)
                        and (not cloud_contribution)
                        and (not total_pressure_contribution)
                    ):
                        if (q == contribution_molecule_species_index) and (
                            i == layer_to_ignore
                        ):
                            n_q = 0.0
                        else:
                            n_q = n_level * X[q, i, j, k]
                    elif total_pressure_contribution and (i == layer_to_ignore):
                        n_q = 0.0
                    else:
                        n_q = n_level * X[q, i, j, k]
                    kappa_Ray[i, j, k, :] += n_q * Rayleigh_stored[q, :]

            if enable_haze == 1:
                for i in range(i_bot, N_layers):
                    for l in range(N_wl):
                        if cloud_contribution and (i == layer_to_ignore):
                            haze_amp = 0.0
                        elif total_pressure_contribution and (i == layer_to_ignore):
                            haze_amp = 0.0
                        else:
                            haze_amp = n[i, j, k] * a * 5.31e-31
                        kappa_cloud[i, j, k, l] += haze_amp * slope[l]

            if enable_deck == 1:
                kappa_cloud[(P > P_cloud[0]), j, k, :] += kappa_cloud_0
                for i in range(i_bot, N_layers):
                    if cloud_contribution and (i == layer_to_ignore):
                        kappa_cloud[i, j, k, :] = 0.0
                    if total_pressure_contribution and (i == layer_to_ignore):
                        kappa_cloud[i, j, k, :] = 0.0

            if enable_surface == 1:
                kappa_gas[(P > P_surf), j, k, :] = 1.0e250

            if enable_Mie == 1:
                aerosol_species_index = 0
                if cloud_contribution and not cloud_total_contribution:
                    for q in range(len(aerosol_species)):
                        if cloud_species == aerosol_species[q]:
                            aerosol_species_index = q

                if len(n_aerosol_array) == len(sigma_Mie_array):
                    for aerosol in range(len(n_aerosol_array)):
                        for i in range(i_bot, N_layers):
                            for l in range(len(wl)):
                                if (
                                    cloud_contribution
                                    and not cloud_total_contribution
                                    and (i == layer_to_ignore)
                                ):
                                    if aerosol != aerosol_species_index:
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k]
                                            * sigma_Mie_array[aerosol][l]
                                        )
                                    else:
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k] * 0.0
                                        )
                                elif cloud_total_contribution and (
                                    i == layer_to_ignore
                                ):
                                    kappa_cloud[i, j, k, l] += (
                                        n_aerosol_array[aerosol][i, j, k] * 0.0
                                    )
                                elif total_pressure_contribution and (
                                    i == layer_to_ignore
                                ):
                                    kappa_cloud[i, j, k, l] += (
                                        n_aerosol_array[aerosol][i, j, k] * 0.0
                                    )
                                else:
                                    kappa_cloud[i, j, k, l] += (
                                        n_aerosol_array[aerosol][i, j, k]
                                        * sigma_Mie_array[aerosol][l]
                                    )
                else:
                    for aerosol in range(len(n_aerosol_array)):
                        if cloud_contribution and cloud_total_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += kappa_cloud_0
                                for i in range(i_bot, N_layers):
                                    if i == layer_to_ignore:
                                        kappa_cloud[i, j, k, :] = 0.0
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        if i == layer_to_ignore:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k] * 0.0
                                            )
                                        else:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k]
                                                * sigma_Mie_array[aerosol - 1][l]
                                            )
                        elif cloud_contribution and not cloud_total_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += kappa_cloud_0
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        if (aerosol - 1 == aerosol_species_index) and (
                                            i == layer_to_ignore
                                        ):
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k] * 0.0
                                            )
                                        else:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k]
                                                * sigma_Mie_array[aerosol - 1][l]
                                            )
                        elif total_pressure_contribution:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += kappa_cloud_0
                                for i in range(i_bot, N_layers):
                                    if i == layer_to_ignore:
                                        kappa_cloud[i, j, k, :] -= kappa_cloud_0
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        if i == layer_to_ignore:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k] * 0.0
                                            )
                                        else:
                                            kappa_cloud[i, j, k, l] += (
                                                n_aerosol_array[aerosol][i, j, k]
                                                * sigma_Mie_array[aerosol - 1][l]
                                            )
                        else:
                            if aerosol == 0:
                                kappa_cloud[(P > P_cloud[0]), j, k, :] += 1.0e250
                            else:
                                for i in range(i_bot, N_layers):
                                    for l in range(len(wl)):
                                        kappa_cloud[i, j, k, l] += (
                                            n_aerosol_array[aerosol][i, j, k]
                                            * sigma_Mie_array[aerosol - 1][l]
                                        )

    return kappa_gas, kappa_Ray, kappa_cloud
