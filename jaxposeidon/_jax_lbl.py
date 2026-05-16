"""JAX-traceable line-by-line (LBL) extinction kernel.

JAX-pure replacement for the inner per-(sector, zone) loop in
``_lbl.compute_kappa_LBL`` (POSEIDON ``absorption.py:1627-1736``).
Unlike the numpy reference (which mutates ``kappa_*`` arrays in
place via per-(j, k) calls), this kernel computes the full
``(N_layers, N_sectors, N_zones, N_wl)`` extinction tensors in a
single vectorised compute.

The HDF5 opacity-table loading (``open_opacity_files``,
``interpolate_sigma_LBL``, ``interpolate_cia_LBL``) stays in
``_lbl.py`` / ``_lbl_table_loader.py`` as setup-only — it cannot be
inside the jit boundary because it does file I/O and produces
shape-dependent arrays. The kernel here takes the pre-interpolated
``sigma_interp`` / ``cia_interp`` tensors as inputs.

``jax.grad`` flows through ``n``, ``X``, ``X_active``, ``X_cia``,
``X_ff``, ``X_bf``, ``sigma_interp``, ``cia_interp``, ``ff_stored``,
``bf_stored``, ``Rayleigh_stored``. The haze/deck/surface boolean
flags and ``enable_*`` scalars are static. ``P_cloud``, ``P_surf``,
``a``, ``gamma``, ``kappa_cloud_0``, ``wl_model`` participate in
ranking/indexing or as continuous parameters (gradient flows through
``a``, ``kappa_cloud_0`` but not through the threshold-comparison
masks against ``P_cloud`` / ``P_surf``).
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


def compute_kappa_LBL_jit(
    wl_model,
    X,
    X_active,
    X_cia,
    X_ff,
    X_bf,
    n,
    P,
    a,
    gamma,
    P_cloud,
    kappa_cloud_0,
    sigma_interp,
    cia_interp,
    Rayleigh_stored,
    ff_stored,
    bf_stored,
    enable_haze,
    enable_deck,
    enable_surface,
    P_surf,
    disable_continuum,
):
    """Pure-jnp kappa_gas / kappa_Ray / kappa_cloud kernel for all
    (sector, zone) at once.

    Shape conventions match POSEIDON ``absorption.py:1627-1736``:
    ``wl_model`` ``(N_wl,)``;
    ``X`` ``(N_species, N_layers, N_sectors, N_zones)``;
    ``X_active`` ``(N_species_active, N_layers, N_sectors, N_zones)``;
    ``X_cia`` ``(2, N_cia_pairs, N_layers, N_sectors, N_zones)``;
    ``X_ff`` ``(2, N_ff_pairs, N_layers, N_sectors, N_zones)``;
    ``X_bf`` ``(N_bf_species, N_layers, N_sectors, N_zones)``;
    ``n`` ``(N_layers, N_sectors, N_zones)``;
    ``P`` ``(N_layers,)``;
    ``sigma_interp`` ``(N_species_active, N_layers, N_wl)``;
    ``cia_interp`` ``(N_cia_pairs, N_layers, N_wl)``;
    ``Rayleigh_stored`` ``(N_species, N_wl)``;
    ``ff_stored`` ``(N_ff_pairs, N_layers, N_wl)``;
    ``bf_stored`` ``(N_bf_species, N_wl)``.

    Returns ``(kappa_gas, kappa_Ray, kappa_cloud)`` each of shape
    ``(N_layers, N_sectors, N_zones, N_wl)``.

    ``enable_haze``, ``enable_deck``, ``enable_surface``,
    ``disable_continuum`` must be static Python ints / bools.
    """
    N_layers = P.shape[0]
    # --- CIA accumulation ---
    # Numerator: n[i,j,k]^2 * X_cia[0,q,i,j,k] * X_cia[1,q,i,j,k]
    n_sq = n * n  # (N_layers, N_s, N_z)
    n_n_cia = n_sq[None, :, :, :] * X_cia[0] * X_cia[1]  # (N_cia, N_layers, N_s, N_z)
    # cia_interp is per-(q, i, l); broadcast to (q, i, j, k, l)
    cia_term = n_n_cia[..., None] * cia_interp[:, :, None, None, :]
    kappa_gas_cia = jnp.sum(cia_term, axis=0)  # (N_layers, N_s, N_z, N_wl)

    # --- Rayleigh accumulation ---
    n_q_Ray = n[None, :, :, :] * X  # (N_species, N_layers, N_s, N_z)
    kappa_Ray = jnp.sum(
        n_q_Ray[..., None] * Rayleigh_stored[:, None, None, None, :], axis=0
    )

    # --- Free-free accumulation (ff_stored is per-layer in LBL) ---
    N_ff = ff_stored.shape[0]
    if N_ff > 0:
        n_n_ff = n_sq[None, :, :, :] * X_ff[0] * X_ff[1]  # (N_ff, N_layers, N_s, N_z)
        ff_term = n_n_ff[..., None] * ff_stored[:, :, None, None, :]
        kappa_gas_ff = jnp.sum(ff_term, axis=0)
    else:
        kappa_gas_ff = jnp.zeros_like(kappa_gas_cia)

    # --- Bound-free accumulation (bf_stored is per-wl, not per-layer) ---
    N_bf = bf_stored.shape[0]
    if N_bf > 0:
        n_q_bf = n[None, :, :, :] * X_bf  # (N_bf, N_layers, N_s, N_z)
        bf_term = n_q_bf[..., None] * bf_stored[:, None, None, None, :]
        kappa_gas_bf = jnp.sum(bf_term, axis=0)
    else:
        kappa_gas_bf = jnp.zeros_like(kappa_gas_cia)

    # --- Active-species accumulation ---
    n_q_active = n[None, :, :, :] * X_active  # (N_act, N_layers, N_s, N_z)
    active_term = n_q_active[..., None] * sigma_interp[:, :, None, None, :]
    kappa_gas_active = jnp.sum(active_term, axis=0)

    # Apply disable_continuum: zero out CIA and Rayleigh
    if disable_continuum:
        kappa_gas_continuum = jnp.zeros_like(kappa_gas_cia)
        kappa_Ray_out = jnp.zeros_like(kappa_Ray)
    else:
        kappa_gas_continuum = kappa_gas_cia
        kappa_Ray_out = kappa_Ray

    kappa_gas = (
        kappa_gas_continuum + kappa_gas_ff + kappa_gas_bf + kappa_gas_active
    )

    # --- Cloud channel ---
    kappa_cloud = jnp.zeros_like(kappa_gas)

    if enable_haze == 1:
        slope = jnp.power(wl_model / 0.35, gamma)  # (N_wl,)
        haze_amp = n * a * 5.31e-31  # (N_layers, N_s, N_z)
        haze_kappa = haze_amp[..., None] * slope[None, None, None, :]
        kappa_cloud = kappa_cloud + haze_kappa

    if enable_deck == 1:
        P_cloud_scalar = jnp.asarray(P_cloud).reshape(())
        deck_mask = (P_cloud_scalar < P).astype(jnp.float64)
        deck_contrib = deck_mask[:, None, None, None] * kappa_cloud_0
        kappa_cloud = kappa_cloud + deck_contrib

    if enable_surface == 1:
        surf_mask = (P_surf < P).astype(jnp.float64)
        kappa_gas = jnp.where(
            surf_mask[:, None, None, None] > 0.5,
            jnp.full_like(kappa_gas, 1.0e250),
            kappa_gas,
        )

    return kappa_gas, kappa_Ray_out, kappa_cloud
