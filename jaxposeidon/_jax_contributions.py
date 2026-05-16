"""JAX-traceable spectral contribution kernel.

JAX-pure replacement for the inner triple-nested loop in
``_contributions.extinction_spectral_contribution`` (POSEIDON
``contributions.py:144-490``). Covers the **active-molecule
contribution** branch (``bulk_species=False``, ``cloud_contribution=False``,
``contribution_species != 'H-'``) — the most common call pattern
from ``spectral_contribution(contribution_species='H2O')``-style
analyses.

The numpy-side selector decisions (string match → species index,
``cia_indices`` list construction) run once **outside** the jit
boundary in ``setup_spectral_contribution_indices`` and produce
fixed-shape integer index arrays / boolean masks. The jit kernel
``spectral_contribution_kernel_jit`` then runs the tensor compute
under jit, and ``jax.grad`` flows through ``X``, ``X_active``,
``X_cia``, ``Rayleigh_stored``, ``cia_stored``, ``sigma_stored``,
``n``, ``T``, ``P``.

Other selector branches (``bulk_species=True``,
``cloud_contribution=True``, ``bound_free=True`` for H-,
``enable_haze``/``enable_deck``/``enable_surface``/``enable_Mie``)
stay on the numpy ``_contributions`` path until lifted in a v1.x
follow-up.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

def _closest_index_jit(value, grid_start, grid_end, N_grid):
    """jit-safe version of ``_opacity_precompute.closest_index``.

    Numerically equivalent to the numpy reference for finite,
    in-range values; clips below/above the grid to the boundary
    indices (matching POSEIDON's Python-branch behaviour). N_grid==1
    is handled explicitly to avoid divide-by-zero on a degenerate
    grid (POSEIDON's branch returns 0 in that case).
    """
    # POSEIDON formula: idx = round((value - grid_start) / dgrid)
    # for uniform grid; clip to [0, N_grid - 1].
    # N_grid == 1 is degenerate (zero-width grid); POSEIDON returns 0.
    if N_grid == 1:
        return jnp.int32(0)
    dgrid = (grid_end - grid_start) / (N_grid - 1)
    idx = jnp.round((value - grid_start) / dgrid).astype(jnp.int32)
    return jnp.clip(idx, 0, N_grid - 1)


def setup_spectral_contribution_indices(
    chemical_species,
    active_species,
    cia_pairs,
    contribution_species,
):
    """Setup-only: resolve string-based selectors to integer indices.

    Returns ``(contribution_species_idx, contribution_active_idx,
    cia_mask, bulk_cia_mask, bulk_species_mask)`` as plain numpy
    arrays. Must run outside ``jit`` because it consumes string
    arrays.
    """
    import numpy as np  # setup-only; allow-listed call path

    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_bulk_species = N_species - N_species_active
    bulk_species_names = chemical_species[:N_bulk_species]

    # Match POSEIDON `_contributions.py:126-127`: both indices default to 0
    # when contribution_species is not found in the respective list.
    # The downstream selectors zero-out the contribution via masks, so
    # the index value at -1/0 is irrelevant for correctness; we use 0
    # to match POSEIDON's silent default and avoid surprising `[-1]`
    # gathers.
    contribution_species_idx = 0
    for i in range(N_species):
        if contribution_species == chemical_species[i]:
            contribution_species_idx = i
    contribution_active_idx = 0
    for i in range(N_species_active):
        if contribution_species == active_species[i]:
            contribution_active_idx = i

    cia_mask = np.zeros(len(cia_pairs), dtype=bool)
    bulk_cia_mask = np.zeros(len(cia_pairs), dtype=bool)
    for i in range(len(cia_pairs)):
        pair_1, pair_2 = cia_pairs[i].split("-")
        if contribution_species == pair_1 or contribution_species == pair_2:
            cia_mask[i] = True
        bulk1 = any(pair_1 == bs for bs in bulk_species_names)
        bulk2 = any(pair_2 == bs for bs in bulk_species_names)
        if bulk1 and bulk2:
            bulk_cia_mask[i] = True

    bulk_species_mask = np.zeros(N_species, dtype=bool)
    bulk_species_mask[:N_bulk_species] = True

    return (
        contribution_species_idx,
        contribution_active_idx,
        cia_mask,
        bulk_cia_mask,
        bulk_species_mask,
    )


def spectral_contribution_kernel_jit(
    n,
    T,
    P,
    X,
    X_active,
    X_cia,
    X_ff,
    sigma_stored,
    cia_stored,
    ff_stored,
    Rayleigh_stored,
    T_fine,
    log_P_fine,
    contribution_species_idx,
    contribution_active_idx,
    cia_combined_mask,
    bulk_species_mask,
    is_He,
    P_deep=1000.0,
):
    """Pure-jnp kappa_gas / kappa_Ray kernel for the active-molecule
    contribution branch.

    Inputs are JAX arrays (or convertible). Selector arrays
    (``cia_combined_mask``, ``bulk_species_mask``,
    ``contribution_*_idx``, ``is_He``) come from
    ``setup_spectral_contribution_indices``. Shape conventions match
    POSEIDON: ``n``, ``T`` ``(N_layers, N_sectors, N_zones)``;
    ``X`` ``(N_species, N_layers, N_sectors, N_zones)``;
    ``X_active`` ``(N_species_active, N_layers, N_sectors, N_zones)``;
    ``X_cia`` ``(2, N_cia_pairs, N_layers, N_sectors, N_zones)``;
    ``X_ff`` ``(2, N_ff_pairs, N_layers, N_sectors, N_zones)``;
    ``sigma_stored`` ``(N_species_active, N_P_fine, N_T_fine, N_wl)``;
    ``cia_stored`` ``(N_cia_pairs, N_T_fine, N_wl)``;
    ``ff_stored`` ``(N_ff_pairs, N_T_fine, N_wl)``;
    ``Rayleigh_stored`` ``(N_species, N_wl)``.

    Returns ``(kappa_gas, kappa_Ray)`` of shape
    ``(N_layers, N_sectors, N_zones, N_wl)``. ``kappa_cloud`` is
    NOT computed here (cloud branches are handled separately in the
    numpy path until the cloud-contribution lift lands).

    Bit-exact with POSEIDON for the active-molecule branch
    (``bulk_species=False``, ``cloud_contribution=False``,
    ``bound_free=False``).

    ``jax.grad`` flows through ``X``, ``X_active``, ``X_cia``,
    ``X_ff``, ``Rayleigh_stored``, ``cia_stored``, ``sigma_stored``,
    ``ff_stored``, and ``n``. It does NOT flow through ``T`` or
    ``P`` — those only feed the integer nearest-index lookups
    (``idx_T_fine``, ``idx_P_fine``, ``i_bot``) which have no
    continuous derivative signal.
    """
    N_layers = P.shape[0]
    N_sectors = n.shape[1]
    N_zones = n.shape[2]
    N_T_fine = T_fine.shape[0]
    N_P_fine = log_P_fine.shape[0]

    # Per-(i,j,k) interpolation indices.
    # closest_index is jit-friendly (pure arithmetic on scalars).
    # vmap over (i,j,k) to produce shape (N_layers, N_sectors, N_zones).
    def _idx_T(T_ijk):
        return _closest_index_jit(T_ijk, T_fine[0], T_fine[-1], N_T_fine)

    def _idx_P_layer(logP_i):
        return _closest_index_jit(logP_i, log_P_fine[0], log_P_fine[-1], N_P_fine)

    idx_T_fine = jax.vmap(jax.vmap(jax.vmap(_idx_T)))(T)  # (N_layers, N_sectors, N_zones)
    idx_P_fine = jax.vmap(_idx_P_layer)(jnp.log10(P))  # (N_layers,)

    # Pressure mask for i_bot cutoff (POSEIDON: i_bot = argmin(|P - P_deep|),
    # then loop runs from i_bot to N_layers — i.e., the **upper** atmosphere
    # above P_deep is summed; below i_bot is zero).
    i_bot = jnp.argmin(jnp.abs(P - P_deep))
    layer_mask = (jnp.arange(N_layers) >= i_bot).astype(jnp.float64)  # (N_layers,)
    layer_mask_4d = layer_mask[:, None, None, None]

    # --- CIA accumulation ---
    # n_n_cia[q, i, j, k] = n[i,j,k]^2 * X_cia[0,q,i,j,k] * X_cia[1,q,i,j,k] * cia_combined_mask[q]
    n_sq = n * n  # (N_layers, N_sectors, N_zones)
    n_n_cia = (
        n_sq[None, :, :, :]
        * X_cia[0]
        * X_cia[1]
        * cia_combined_mask[:, None, None, None].astype(jnp.float64)
    )  # (N_cia_pairs, N_layers, N_sectors, N_zones)
    # cia_at_idx[q, i, j, k, l] = cia_stored[q, idx_T_fine[i,j,k], l]
    # cia_stored shape: (N_cia_pairs, N_T_fine, N_wl).
    # Use explicit advanced indexing to get exact shape (N_cia, N_layers, N_s, N_z, N_wl).
    N_cia = cia_stored.shape[0]
    q_idx = jnp.arange(N_cia)[:, None, None, None]  # (N_cia, 1, 1, 1)
    t_idx = idx_T_fine[None, :, :, :]  # (1, N_layers, N_sectors, N_zones)
    cia_at_idx = cia_stored[q_idx, t_idx, :]  # (N_cia, N_layers, N_s, N_z, N_wl)
    kappa_gas_cia = jnp.sum(n_n_cia[..., None] * cia_at_idx, axis=0)  # (N_layers, N_s, N_z, N_wl)

    # --- Free-free accumulation (POSEIDON _contributions.py:184-188) ---
    # n_n_ff[q, i, j, k] = n[i,j,k]^2 * X_ff[0,q,i,j,k] * X_ff[1,q,i,j,k]
    # — unconditionally added (no selector); zero-N_ff_pairs is handled
    # by jnp.sum over an empty axis returning 0.
    N_ff = ff_stored.shape[0]
    if N_ff > 0:
        n_n_ff = n_sq[None, :, :, :] * X_ff[0] * X_ff[1]  # (N_ff_pairs, N_layers, N_s, N_z)
        q_idx_ff = jnp.arange(N_ff)[:, None, None, None]
        ff_at_idx = ff_stored[q_idx_ff, t_idx, :]  # (N_ff_pairs, N_layers, N_s, N_z, N_wl)
        kappa_gas_ff = jnp.sum(n_n_ff[..., None] * ff_at_idx, axis=0)
    else:
        kappa_gas_ff = jnp.zeros_like(kappa_gas_cia)

    # --- Active-species accumulation ---
    # Only the species at contribution_active_idx contributes (and not when is_He).
    # n_q[i,j,k] = n[i,j,k] * X_active[contribution_active_idx, i, j, k]
    n_active = n * X_active[contribution_active_idx]  # (N_layers, N_sectors, N_zones)
    sigma_at_idx_P = jnp.take(
        sigma_stored[contribution_active_idx], idx_P_fine, axis=0
    )  # (N_layers, N_T_fine, N_wl)

    # Gather along T axis using idx_T_fine[i, j, k] via explicit indexing:
    # sigma_at_PT[i, j, k, l] = sigma_at_idx_P[i, idx_T_fine[i, j, k], l]
    sigma_at_PT = sigma_at_idx_P[
        jnp.arange(N_layers)[:, None, None],
        idx_T_fine,
        :,
    ]  # (N_layers, N_sectors, N_zones, N_wl)

    # Active contribution: only adds when not is_He
    kappa_gas_active = (1.0 - jnp.asarray(is_He, dtype=jnp.float64)) * (
        n_active[..., None] * sigma_at_PT
    )

    kappa_gas = (kappa_gas_cia + kappa_gas_ff + kappa_gas_active) * layer_mask_4d

    # --- Rayleigh accumulation ---
    # For the active-molecule branch: contribution species + bulk species both add.
    # rayleigh_mask[q] = (q == contribution_species_idx) OR bulk_species_mask[q]
    rayleigh_mask = (
        (jnp.arange(X.shape[0]) == contribution_species_idx)
        | bulk_species_mask
    ).astype(jnp.float64)
    # n_q[q, i, j, k] = n[i,j,k] * X[q,i,j,k] * rayleigh_mask[q]
    n_q_Ray = n[None, :, :, :] * X * rayleigh_mask[:, None, None, None]
    # Rayleigh_stored: (N_species, N_wl)
    kappa_Ray = jnp.sum(
        n_q_Ray[..., None] * Rayleigh_stored[:, None, None, None, :], axis=0
    ) * layer_mask_4d

    return kappa_gas, kappa_Ray
