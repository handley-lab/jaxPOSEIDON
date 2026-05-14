"""Line-by-line extinction kernel.

Ports POSEIDON `absorption.py:1627-1736` (`compute_kappa_LBL`). This
is the per-(sector, zone) kernel that accumulates kappa_gas, kappa_Ray,
kappa_cloud after the caller has interpolated cross-sections to per-layer
(P, T) using the LBL HDF5 grids (loaded via
`_lbl_table_loader.py`). The full `extinction_LBL` orchestrator
(`absorption.py:1739-1951`) is the follow-up — it requires the real
~10 GB POSEIDON opacity HDF5 files (env-gated).
"""

import numpy as np


def compute_kappa_LBL(
    j,
    k,
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
    N_species,
    N_species_active,
    N_cia_pairs,
    N_ff_pairs,
    N_bf_species,
    sigma_interp,
    cia_interp,
    Rayleigh_stored,
    ff_stored,
    bf_stored,
    enable_haze,
    enable_deck,
    enable_surface,
    kappa_gas,
    kappa_Ray,
    kappa_cloud,
    P_surf,
    disable_continuum,
):
    """Accumulate kappa_gas/Ray/cloud for sector j, zone k (LBL mode).

    Bit-equivalent port of POSEIDON `absorption.py:1627-1736`. Mutates
    `kappa_gas`, `kappa_Ray`, `kappa_cloud` in place; the caller is
    expected to have pre-allocated them shape (N_layers, N_sectors,
    N_zones, N_wl).
    """
    N_wl = len(wl_model)
    N_layers = len(P)

    if enable_haze == 1:
        slope = np.power((wl_model / 0.35), gamma)

    for i in range(N_layers):
        if not disable_continuum:
            for q in range(N_cia_pairs):
                n_cia_1 = n[i, j, k] * X_cia[0, q, i, j, k]
                n_cia_2 = n[i, j, k] * X_cia[1, q, i, j, k]
                n_n_cia = n_cia_1 * n_cia_2
                for l in range(N_wl):
                    kappa_gas[i, j, k, l] += n_n_cia * cia_interp[q, i, l]

            for q in range(N_species):
                n_q = n[i, j, k] * X[q, i, j, k]
                for l in range(N_wl):
                    kappa_Ray[i, j, k, l] += n_q * Rayleigh_stored[q, l]

        for q in range(N_ff_pairs):
            n_ff_1 = n[i, j, k] * X_ff[0, q, i, j, k]
            n_ff_2 = n[i, j, k] * X_ff[1, q, i, j, k]
            n_n_ff = n_ff_1 * n_ff_2
            for l in range(N_wl):
                kappa_gas[i, j, k, l] += n_n_ff * ff_stored[q, i, l]

        for q in range(N_bf_species):
            n_q = n[i, j, k] * X_bf[q, i, j, k]
            for l in range(N_wl):
                kappa_gas[i, j, k, l] += n_q * bf_stored[q, l]

        for q in range(N_species_active):
            n_q = n[i, j, k] * X_active[q, i, j, k]
            for l in range(N_wl):
                kappa_gas[i, j, k, l] += n_q * sigma_interp[q, i, l]

    if enable_haze == 1:
        for i in range(N_layers):
            haze_amp = n[i, j, k] * a * 5.31e-31
            for l in range(N_wl):
                kappa_cloud[i, j, k, l] += haze_amp * slope[l]

    if enable_deck == 1:
        kappa_cloud[(P_cloud < P), j, k, :] += kappa_cloud_0

    if enable_surface == 1:
        kappa_gas[(P_surf < P), j, k, :] = 1.0e250
