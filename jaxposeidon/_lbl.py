"""Line-by-line extinction (kernel + orchestrator).

Ports POSEIDON `absorption.py:204-236, 1386-1593, 1627-1951`:
- `T_interpolation_init` (temperature interpolation index/weight setup),
- `interpolate_cia_LBL` (CIA cross-section interpolation to layer T),
- `interpolate_sigma_LBL` (per-(P,T) molecular cross-section interpolation),
- `compute_kappa_LBL` (per-(sector, zone) extinction accumulator),
- `extinction_LBL` (orchestrator that opens HDF5 opacity tables, drives
  the per-species/per-pair interpolation, and assembles kappa).
"""

import numpy as np

from jaxposeidon._h_minus import H_minus_bound_free, H_minus_free_free
from jaxposeidon._lbl_table_loader import open_opacity_files
from jaxposeidon._opacity_precompute import (
    closest_index,
    prior_index,
    prior_index_V2,
)


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


def T_interpolation_init(N_T_fine, T_grid, T_fine, y):
    """Precompute T-grid interpolation weights for each fine T point.

    Port of POSEIDON `absorption.py:204-236`. Mutates `y` in place; the
    sentinel values `-1`/`-2` flag fine temperatures off the LHS/RHS of
    the opacity T grid (handled by the per-point interpolators).
    """
    w_T = np.zeros(N_T_fine)

    for j in range(N_T_fine):
        if T_fine[j] < T_grid[0]:
            y[j] = -1
            w_T[j] = 0.0
        elif T_fine[j] >= T_grid[-1]:
            y[j] = -2
            w_T[j] = 0.0
        else:
            y[j] = prior_index(T_fine[j], T_grid, 0)
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j] + 1]
            w_T[j] = 1.0 / ((1.0 / T2) - (1.0 / T1))

    return w_T


def interpolate_cia_LBL(
    P, log_cia, nu_model, nu_cia, T, T_grid_cia, N_T_cia, N_wl, N_nu, y, w_T
):
    """Interpolate a CIA binary cross section onto layer T (LBL mode).

    Port of POSEIDON `absorption.py:1386-1434`.
    """
    N_layers = len(P)
    cia_inp = np.zeros(shape=(N_layers, N_wl))
    N_nu_cia = len(nu_cia)

    for i in range(N_layers):
        T_i = T[i]
        T1 = T_grid_cia[y[i]]
        T2 = T_grid_cia[y[i] + 1]

        for k in range(N_nu):
            z = closest_index(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia)

            if (z == 0) or (z == (N_nu_cia - 1)):
                cia_inp[i, (N_wl - 1) - k] = 0.0
            else:
                # log_cia is float32 (matches POSEIDON's `.astype(float32)`
                # at the read site). numba's nopython `int_literal **
                # float32_scalar` is promoted to float64; plain numpy
                # `10 ** float32_scalar` stays float32 (and underflows
                # for log_cia ~< -38). Cast to Python float64 explicitly
                # at scalar-index sites to mirror numba's promotion. The
                # interior branch uses the array-slice form (float32),
                # matching POSEIDON's identical pattern.
                if y[i] == -1:
                    cia_inp[i, (N_wl - 1) - k] = 10 ** float(log_cia[0, z])
                elif y[i] == -2:
                    cia_inp[i, (N_wl - 1) - k] = 10 ** float(log_cia[N_T_cia - 1, z])
                else:
                    cia_reduced = 10 ** log_cia[y[i] : y[i] + 2, z]
                    cia_1, cia_2 = cia_reduced[0], cia_reduced[1]
                    cia_inp[i, (N_wl - 1) - k] = np.power(
                        cia_1, w_T[i] * ((1.0 / T2) - (1.0 / T_i))
                    ) * np.power(cia_2, w_T[i] * ((1.0 / T_i) - (1.0 / T1)))

    return cia_inp


def interpolate_sigma_LBL(
    log_sigma,
    nu_model,
    nu_opac,
    P,
    T,
    log_P_grid,
    T_grid,
    N_T,
    N_P,
    N_wl,
    N_nu,
    y,
    w_T,
):
    """Interpolate a molecular cross section onto layer (P, T) (LBL mode).

    Port of POSEIDON `absorption.py:1438-1593`.
    """
    N_layers = len(P)
    sigma_inp = np.zeros(shape=(N_layers, N_wl))

    nu_opac_min = nu_opac[0]
    nu_opac_max = nu_opac[-1]
    N_nu_opac = len(nu_opac)

    nu_model_min = nu_model[0]
    nu_model_max = nu_model[-1]

    z_grid_min = closest_index(nu_model_min, nu_opac[0], nu_opac[-1], N_nu_opac)
    z_grid_max = closest_index(nu_model_max, nu_opac[0], nu_opac[-1], N_nu_opac)

    reduced_log_sigma = log_sigma[:, :, z_grid_min : z_grid_max + 1]

    log_P = np.log10(P)

    x = np.zeros(N_layers).astype(np.int64)
    w_P = np.zeros(N_layers)
    b1 = np.zeros(shape=(N_layers))
    b2 = np.zeros(shape=(N_layers))

    for i in range(N_layers):
        if log_P[i] < log_P_grid[0]:
            x[i] = -1
            w_P[i] = 0.0
        elif log_P[i] >= log_P_grid[-1]:
            x[i] = -2
            w_P[i] = 0.0
        else:
            x[i] = prior_index_V2(log_P[i], log_P_grid[0], log_P_grid[-1], N_P)
            w_P[i] = (log_P[i] - log_P_grid[x[i]]) / (
                log_P_grid[x[i] + 1] - log_P_grid[x[i]]
            )

        b1[i] = 1.0 - w_P[i]
        b2[i] = w_P[i]

    for i in range(N_layers):
        T_i = T[i]
        T1 = T_grid[y[i]]
        T2 = T_grid[y[i] + 1]

        for k in range(N_nu):
            nu_model_k = nu_model[k]

            if (nu_model_k < nu_opac_min) or (nu_model_k > nu_opac_max):
                sigma_inp[i, (N_wl - 1) - k] = 0.0
            else:
                log_sigma_PT_rectangle = reduced_log_sigma[
                    x[i] : x[i] + 2, y[i] : y[i] + 2, k
                ]

                # Scalar-index sites cast to float to mirror numba's
                # nopython float64 promotion of `10 ** float32_scalar`.
                # Where POSEIDON uses an array-rectangle slice, the
                # float32 expression is preserved verbatim.
                if x[i] == -1:
                    if y[i] == -1:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** float(
                            reduced_log_sigma[0, 0, k]
                        )
                    elif y[i] == -2:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** float(
                            reduced_log_sigma[0, N_T - 1, k]
                        )
                    else:
                        sig_T1 = 10 ** float(reduced_log_sigma[0, y[i], k])
                        sig_T2 = 10 ** float(reduced_log_sigma[0, y[i] + 1, k])
                        sigma_inp[i, (N_wl - 1) - k] = np.power(
                            sig_T1, (w_T[i] * ((1.0 / T2) - (1.0 / T_i)))
                        ) * np.power(sig_T2, (w_T[i] * ((1.0 / T_i) - (1.0 / T1))))
                elif x[i] == -2:
                    if y[i] == -1:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** float(
                            reduced_log_sigma[N_P - 1, 0, k]
                        )
                    elif y[i] == -2:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** float(
                            reduced_log_sigma[N_P - 1, N_T - 1, k]
                        )
                    else:
                        sig_T1 = 10 ** float(reduced_log_sigma[N_P - 1, y[i], k])
                        sig_T2 = 10 ** float(reduced_log_sigma[N_P - 1, y[i] + 1, k])
                        sigma_inp[i, (N_wl - 1) - k] = np.power(
                            sig_T1, (w_T[i] * ((1.0 / T2) - (1.0 / T_i)))
                        ) * np.power(sig_T2, (w_T[i] * ((1.0 / T_i) - (1.0 / T1))))
                else:
                    if y[i] == -1:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** (
                            b1[i] * reduced_log_sigma[x[i], 0, k]
                            + b2[i] * reduced_log_sigma[x[i] + 1, 0, k]
                        )
                    elif y[i] == -2:
                        sigma_inp[i, (N_wl - 1) - k] = 10 ** (
                            b1[i] * reduced_log_sigma[x[i], N_T - 1, k]
                            + b2[i] * reduced_log_sigma[x[i] + 1, N_T - 1, k]
                        )
                    else:
                        sig_T1 = 10 ** (
                            b1[i] * (log_sigma_PT_rectangle[0, 0])
                            + b2[i] * (log_sigma_PT_rectangle[1, 0])
                        )
                        sig_T2 = 10 ** (
                            b1[i] * (log_sigma_PT_rectangle[0, 1])
                            + b2[i] * (log_sigma_PT_rectangle[1, 1])
                        )
                        sigma_inp[i, (N_wl - 1) - k] = np.power(
                            sig_T1, (w_T[i] * ((1.0 / T2) - (1.0 / T_i)))
                        ) * np.power(sig_T2, (w_T[i] * ((1.0 / T_i) - (1.0 / T1))))

    return sigma_inp


def extinction_LBL(
    chemical_species,
    active_species,
    cia_pairs,
    ff_pairs,
    bf_species,
    n,
    T,
    P,
    wl_model,
    X,
    X_active,
    X_cia,
    X_ff,
    X_bf,
    a,
    gamma,
    P_cloud,
    kappa_cloud_0,
    Rayleigh_stored,
    enable_haze,
    enable_deck,
    enable_surface,
    N_sectors,
    N_zones,
    P_surf,
    opacity_database="High-T",
    disable_continuum=False,
    suppress_print=False,
    database_version="1.3",
):
    """Line-by-line extinction-coefficient orchestrator.

    Port of POSEIDON `absorption.py:1739-1951`. Opens the HDF5 opacity
    tables via `open_opacity_files`, drives the per-(P, T) interpolation
    via `interpolate_cia_LBL` / `interpolate_sigma_LBL` and the H-minus
    free-free / bound-free fits, then accumulates kappa via
    `compute_kappa_LBL`.

    Note: POSEIDON closes the CIA and opacity HDF5 files inside the
    `(N_sectors, N_zones)` loop body (a known upstream quirk); this port
    preserves the same behaviour for bit-equivalence. As a result the
    LBL path supports only `N_sectors == N_zones == 1` until upstream
    moves the closes outside the loop.
    """
    if not suppress_print:
        print("Reading in cross sections in line-by-line mode...")

    N_species = len(chemical_species)
    N_species_active = len(active_species)
    N_cia_pairs = len(cia_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)
    N_layers = len(P)

    nu_model = 1.0e4 / wl_model
    nu_model = nu_model[::-1]

    N_nu = len(nu_model)
    N_wl = len(wl_model)

    kappa_gas = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros(shape=(N_layers, N_sectors, N_zones, N_wl))

    opac_file, cia_file = open_opacity_files(opacity_database, database_version)

    for j in range(N_sectors):
        for k in range(N_zones):
            cia_interp = np.zeros(shape=(N_cia_pairs, N_layers, N_wl))

            for q in range(N_cia_pairs):
                cia_pair_q = cia_pairs[q]

                T_grid_cia_q = np.array(cia_file[cia_pair_q + "/T"])
                nu_cia_q = np.array(cia_file[cia_pair_q + "/nu"])
                N_T_cia_q = len(T_grid_cia_q)

                y_cia_q = np.zeros(N_layers, dtype=np.int64)
                w_T_cia_q = T_interpolation_init(
                    N_layers, T_grid_cia_q, T[:, j, k], y_cia_q
                )

                log_cia_q = np.array(cia_file[cia_pair_q + "/log(cia)"]).astype(
                    np.float32
                )

                cia_interp[q, :, :] = interpolate_cia_LBL(
                    P,
                    log_cia_q,
                    nu_model,
                    nu_cia_q,
                    T[:, j, k],
                    T_grid_cia_q,
                    N_T_cia_q,
                    N_wl,
                    N_nu,
                    y_cia_q,
                    w_T_cia_q,
                )

                del log_cia_q, nu_cia_q, w_T_cia_q, y_cia_q

                if not suppress_print:
                    print(cia_pair_q + " done")

            cia_file.close()

            ff_stored = np.zeros(shape=(N_ff_pairs, N_layers, N_wl))

            for q in range(N_ff_pairs):
                ff_pair_q = ff_pairs[q]

                if ff_pair_q == "H-ff":
                    ff_stored[q, :, :] = H_minus_free_free(wl_model, T[:, j, k])
                else:
                    raise Exception("Unsupported free-free opacity.")

                if not suppress_print:
                    print(ff_pair_q + " done")

            bf_stored = np.zeros(shape=(N_bf_species, N_wl))

            for q in range(N_bf_species):
                bf_species_q = bf_species[q]

                if bf_species_q == "H-bf":
                    bf_stored[q, :] = H_minus_bound_free(wl_model)
                else:
                    raise Exception("Unsupported bound-free opacity.")

                if not suppress_print:
                    print(bf_species_q + " done")

            sigma_interp = np.zeros(shape=(N_species_active, N_layers, N_wl))

            for q in range(N_species_active):
                species_q = active_species[q]

                T_grid_q = np.array(opac_file[species_q + "/T"])
                log_P_grid_q = np.array(opac_file[species_q + "/log(P)"])
                nu_q = np.array(opac_file[species_q + "/nu"])

                N_T_q = len(T_grid_q)
                N_P_q = len(log_P_grid_q)

                y_q = np.zeros(N_layers, dtype=np.int64)
                w_T_q = T_interpolation_init(N_layers, T_grid_q, T[:, j, k], y_q)

                log_sigma_q = np.array(opac_file[species_q + "/log(sigma)"]).astype(
                    np.float32
                )

                sigma_interp[q, :, :] = interpolate_sigma_LBL(
                    log_sigma_q,
                    nu_model,
                    nu_q,
                    P,
                    T[:, j, k],
                    log_P_grid_q,
                    T_grid_q,
                    N_T_q,
                    N_P_q,
                    N_wl,
                    N_nu,
                    y_q,
                    w_T_q,
                )

                del log_sigma_q, nu_q, w_T_q, y_q

                if not suppress_print:
                    print(species_q + " done")

            opac_file.close()

            compute_kappa_LBL(
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
            )

    if not suppress_print:
        print("Finished producing extinction coefficients")

    return kappa_gas, kappa_Ray, kappa_cloud
