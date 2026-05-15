"""Runtime extinction — assemble κ from pre-interpolated cross sections.

Faithful port of the v0 subset of POSEIDON `absorption.py:1034-1227`
(`extinction(...)`). Mirrors POSEIDON's nearest-fine-grid lookup in
(log_P, T) and the MacMad17 deck + haze additions.

v1-B: rewritten in `jax.numpy` with `jnp.where`-based masking over
the `i_bot:N_layers` slice so the function is jit-traceable. The
outer (`j`, `k`) sector/zone Python loops unroll at trace time
(static loop bounds from input shapes); per-layer accumulation uses
broadcasted vectorisation.

v0 envelope:
- enable_haze, enable_deck, enable_surface, enable_Mie in {0, 1}
  (Python-static integers passed as static arguments to jit)
- N_ff_pairs, N_bf_species derived from `ff_pairs`/`bf_species`
  static lengths
- N_sectors = N_zones = 1 for the K2-18b use case (multi-(j, k)
  loops unroll at trace time when needed)

CIA and active-species opacity are the heaviest branches and are
ported in full. The output 4-tuple matches POSEIDON's:
    (kappa_gas, kappa_Ray, kappa_cloud, kappa_cloud_separate)
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxposeidon._opacity_precompute import (  # noqa: E402
    closest_index,
    closest_index_jax,
)

__all__ = ["extinction", "closest_index"]


def extinction(
    chemical_species,
    active_species,
    cia_pairs,
    ff_pairs,
    bf_species,
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
    disable_continuum=False,
):
    """Compute kappa_gas, kappa_Ray, kappa_cloud arrays.

    Bit-exact port of POSEIDON `absorption.py:1034-1227` for the v0
    envelope plus the Phase 0.5.13d surface branch and Phase 0.5.12b
    Mie aerosol branches.
    """
    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)
    N_wl = len(wl)
    N_layers = len(P)

    T = jnp.asarray(T, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)
    P = jnp.asarray(P, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    X = jnp.asarray(X, dtype=jnp.float64)
    X_active = jnp.asarray(X_active, dtype=jnp.float64)
    X_cia = jnp.asarray(X_cia, dtype=jnp.float64)
    X_ff = jnp.asarray(X_ff, dtype=jnp.float64)
    X_bf = jnp.asarray(X_bf, dtype=jnp.float64)
    sigma_stored = jnp.asarray(sigma_stored, dtype=jnp.float64)
    cia_stored = jnp.asarray(cia_stored, dtype=jnp.float64)
    Rayleigh_stored = jnp.asarray(Rayleigh_stored, dtype=jnp.float64)
    if N_ff_pairs > 0:
        ff_stored = jnp.asarray(ff_stored, dtype=jnp.float64)
    if N_bf_species > 0:
        bf_stored = jnp.asarray(bf_stored, dtype=jnp.float64)

    N_T_fine = len(T_fine)
    N_P_fine = len(log_P_fine)
    T_fine_start = float(T_fine[0])
    T_fine_end = float(T_fine[-1])
    log_P_fine_start = float(log_P_fine[0])
    log_P_fine_end = float(log_P_fine[-1])

    i_bot_data = jnp.argmin(jnp.abs(P - P_deep))
    layer_idx = jnp.arange(N_layers)
    layer_mask = layer_idx >= i_bot_data  # (N_layers,)

    P_arr = jnp.asarray(P_cloud, dtype=jnp.float64)

    kappa_gas = jnp.zeros((N_layers, N_sectors, N_zones, N_wl), dtype=jnp.float64)
    kappa_Ray = jnp.zeros((N_layers, N_sectors, N_zones, N_wl), dtype=jnp.float64)
    kappa_cloud = jnp.zeros((N_layers, N_sectors, N_zones, N_wl), dtype=jnp.float64)
    N_aer = len(n_aerosol_array) if enable_Mie == 1 else 0
    kappa_cloud_separate = jnp.zeros(
        (N_aer, N_layers, N_sectors, N_zones, N_wl), dtype=jnp.float64
    )

    if enable_haze == 1:
        slope = jnp.power(wl / 0.35, gamma)

    log_P = jnp.log10(P)

    def _idx_T(T_jk):
        return closest_index_jax(T_jk, T_fine_start, T_fine_end, N_T_fine)

    def _idx_P(log_P_i):
        return closest_index_jax(log_P_i, log_P_fine_start, log_P_fine_end, N_P_fine)

    for j in range(N_sectors):
        for k in range(N_zones):
            for i in range(N_layers):
                n_level = n[i, j, k]
                idx_T_fine = _idx_T(T[i, j, k])
                idx_P_fine = _idx_P(log_P[i])

                contrib = jnp.zeros((N_wl,), dtype=jnp.float64)

                if not disable_continuum:
                    for q in range(N_cia_pairs):
                        n_cia_1 = n_level * X_cia[0, q, i, j, k]
                        n_cia_2 = n_level * X_cia[1, q, i, j, k]
                        n_n_cia = n_cia_1 * n_cia_2
                        contrib = contrib + n_n_cia * cia_stored[q, idx_T_fine, :]

                for q in range(N_ff_pairs):
                    n_ff_1 = n_level * X_ff[0, q, i, j, k]
                    n_ff_2 = n_level * X_ff[1, q, i, j, k]
                    n_n_ff = n_ff_1 * n_ff_2
                    contrib = contrib + n_n_ff * ff_stored[q, idx_T_fine, :]

                for q in range(N_bf_species):
                    n_q = n_level * X_bf[q, i, j, k]
                    contrib = contrib + n_q * bf_stored[q, :]

                for q in range(N_species_active):
                    n_q = n_level * X_active[q, i, j, k]
                    contrib = contrib + (
                        n_q * sigma_stored[q, idx_P_fine, idx_T_fine, :]
                    )

                mask = layer_mask[i]
                kappa_gas = kappa_gas.at[i, j, k, :].set(jnp.where(mask, contrib, 0.0))

                if not disable_continuum:
                    ray = jnp.zeros((N_wl,), dtype=jnp.float64)
                    for q in range(N_species):
                        n_q = n_level * X[q, i, j, k]
                        ray = ray + n_q * Rayleigh_stored[q, :]
                    kappa_Ray = kappa_Ray.at[i, j, k, :].set(jnp.where(mask, ray, 0.0))

            if enable_haze == 1:
                for i in range(N_layers):
                    haze_amp = n[i, j, k] * a * 5.31e-31
                    increment = haze_amp * slope
                    kappa_cloud = kappa_cloud.at[i, j, k, :].add(
                        jnp.where(layer_mask[i], increment, 0.0)
                    )

            if enable_deck == 1:
                deck_mask = P_arr[0] < P  # (N_layers,)
                kappa_cloud = kappa_cloud.at[:, j, k, :].add(
                    deck_mask[:, None].astype(jnp.float64) * kappa_cloud_0
                )

            if enable_surface == 1:
                surf_mask = P_surf < P
                kappa_gas = kappa_gas.at[:, j, k, :].set(
                    jnp.where(surf_mask[:, None], 1.0e250, kappa_gas[:, j, k, :])
                )

            if enable_Mie == 1:
                if N_aer == len(sigma_Mie_array):
                    for aer in range(N_aer):
                        n_aer = jnp.asarray(n_aerosol_array[aer], dtype=jnp.float64)
                        sig_aer = jnp.asarray(sigma_Mie_array[aer], dtype=jnp.float64)
                        for i in range(N_layers):
                            inc = jnp.where(
                                layer_mask[i], n_aer[i, j, k] * sig_aer, 0.0
                            )
                            kappa_cloud = kappa_cloud.at[i, j, k, :].add(inc)
                            kappa_cloud_separate = kappa_cloud_separate.at[
                                aer, i, j, k, :
                            ].add(inc)
                else:
                    for aer in range(N_aer):
                        n_aer = jnp.asarray(n_aerosol_array[aer], dtype=jnp.float64)
                        if aer == 0:
                            deck_mask = P_arr[0] < P
                            inc_deck = deck_mask[:, None].astype(jnp.float64) * 1.0e250
                            kappa_cloud = kappa_cloud.at[:, j, k, :].add(inc_deck)
                            kappa_cloud_separate = kappa_cloud_separate.at[
                                aer, :, j, k, :
                            ].add(inc_deck)
                        else:
                            sig_aer = jnp.asarray(
                                sigma_Mie_array[aer - 1], dtype=jnp.float64
                            )
                            for i in range(N_layers):
                                inc = jnp.where(
                                    layer_mask[i], n_aer[i, j, k] * sig_aer, 0.0
                                )
                                kappa_cloud = kappa_cloud.at[i, j, k, :].add(inc)
                                kappa_cloud_separate = kappa_cloud_separate.at[
                                    aer, i, j, k, :
                                ].add(inc)

    return kappa_gas, kappa_Ray, kappa_cloud, kappa_cloud_separate
