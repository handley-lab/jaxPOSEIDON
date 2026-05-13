"""Atmosphere column construction for v0.

Mirrors the v0 paths through POSEIDON's `atmosphere.py:profiles(...)`
(`POSEIDON/atmosphere.py:2015-2493`). v0 envelope:

- `PT_profile in {'isotherm', 'Madhu'}`, `PT_dim == 1`
- `X_profile == 'isochem'`, `X_dim == 1`, no per-species gradients
- `bulk_species` is either `['H2']`, `['H2', 'He']`, or any single non-ghost
  species. H2+H+He dissociation mixture is deferred.
- `Atmosphere_dimension == 1` (N_sectors = N_zones = 1)
- `disable_atmosphere == False`

Numerical strategy: this port mirrors POSEIDON's algorithms exactly,
including its float64 numpy + scipy `gaussian_filter1d(sigma=3,
mode='nearest')` smoothing for the Madhu profile. The smoothing is
delegated to scipy so we get bit-equivalent output to POSEIDON. A
JAX-native gaussian smoother is a v1 work item if `jax.grad` through the
PT parameters is needed.
"""

import numpy as np
import scipy.constants as sc
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# P-T profile builders (1D only)
# ---------------------------------------------------------------------------
def compute_T_isotherm(P, T_iso):
    """1D isothermal profile, shape (N_layers, 1, 1).

    Mirrors `POSEIDON/atmosphere.py:2152-2161` (no Gaussian smoothing for
    isotherm).
    """
    return T_iso * np.ones(shape=(len(P), 1, 1))


def compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_set, P_set):
    """1D Madhusudhan & Seager (2009) P-T profile.

    Bit-equivalent to `POSEIDON/atmosphere.py:20-103`. Caller should
    Gaussian-smooth the output with `gauss_conv(...)` to match POSEIDON
    `atmosphere.py:2262`.
    """
    N_layers = len(P)
    T = np.zeros(shape=(N_layers, 1, 1))

    i_set = int(np.argmin(np.abs(P - P_set)))
    P_set_i = P[i_set]

    log_P = np.log10(P)
    log_P_min = np.log10(np.min(P))
    log_P_set_i = np.log10(P_set_i)

    # Boundary temperatures determined by which layer contains P_set.
    if log_P_set_i >= log_P3:
        T3 = T_set
        T2 = T3 - ((1.0 / a2) * (log_P3 - log_P2)) ** 2
        T1 = T2 + ((1.0 / a2) * (log_P1 - log_P2)) ** 2
        T0 = T1 - ((1.0 / a1) * (log_P1 - log_P_min)) ** 2
    elif log_P_set_i >= log_P1:
        T2 = T_set - ((1.0 / a2) * (log_P_set_i - log_P2)) ** 2
        T1 = T2 + ((1.0 / a2) * (log_P1 - log_P2)) ** 2
        T3 = T2 + ((1.0 / a2) * (log_P3 - log_P2)) ** 2
        T0 = T1 - ((1.0 / a1) * (log_P1 - log_P_min)) ** 2
    else:  # log_P_set_i < log_P1
        T0 = T_set - ((1.0 / a1) * (log_P_set_i - log_P_min)) ** 2
        T1 = T0 + ((1.0 / a1) * (log_P1 - log_P_min)) ** 2
        T2 = T1 - ((1.0 / a2) * (log_P1 - log_P2)) ** 2
        T3 = T2 + ((1.0 / a2) * (log_P3 - log_P2)) ** 2

    for i in range(N_layers):
        if log_P[i] >= log_P3:
            T[i, 0, 0] = T3
        elif log_P[i] > log_P1 and log_P[i] < log_P3:
            T[i, 0, 0] = T2 + ((1.0 / a2) * (log_P[i] - log_P2)) ** 2
        else:  # log_P[i] <= log_P1
            T[i, 0, 0] = T0 + ((1.0 / a1) * (log_P[i] - log_P_min)) ** 2

    return T


def gauss_conv(arr, sigma=3, axis=0, mode="nearest"):
    """Gaussian-smooth `arr` along `axis`, matching POSEIDON's scipy call.

    POSEIDON does `from scipy.ndimage import gaussian_filter1d as gauss_conv`
    (`POSEIDON/atmosphere.py:9`) and applies it as
    `gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')` for Madhu/gradient
    profiles. Defaults match.
    """
    return gaussian_filter1d(arr, sigma=sigma, axis=axis, mode=mode)


# ---------------------------------------------------------------------------
# Mixing-ratio construction (isochem, 1D)
# ---------------------------------------------------------------------------
def compute_X_isochem_1D(P, log_X_state, N_sectors, N_zones, param_species):
    """v0 isochem path through compute_X_field_gradient
    (`POSEIDON/atmosphere.py:759-873`).

    With X_dim=1 and no per-species gradients,
    `log_X_state[q] = [log_X_iso, 0, 0, log_X_iso]` and the gradient
    function collapses to constant `X = 10**log_X_iso` across all layers,
    sectors, and zones.
    """
    N_layers = len(P)
    N_param_species = len(param_species)
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))
    for q in range(N_param_species):
        log_X_bar_term, _Delta_term, _Delta_DN, _log_X_deep = log_X_state[q, :]
        X = 10.0 ** log_X_bar_term
        X_profiles[q, :, :, :] = X
    return X_profiles


def add_bulk_component(P, T, X_param, N_species, N_sectors, N_zones,
                      bulk_species, He_fraction):
    """Concatenate bulk species mixing ratios so the columns sum to 1.

    Mirrors `POSEIDON/atmosphere.py:1221-1330` for the v0-supported
    cases:
      - ['H2', 'He']  (H2/He bulk, fixed He/H2)
      - ['H', 'He']   (H/He bulk)
      - single non-ghost species (e.g. ['N2'])
    H2/H/He dissociation mixture is deferred (would require
    Parmentier_dissociation_profile).
    """
    N_layers = len(P)
    N_bulk_species = len(bulk_species)
    X = np.zeros(shape=(N_species, N_layers, N_sectors, N_zones))

    bulk_set = set(bulk_species)
    if {"H2", "He"} <= bulk_set and "H" not in bulk_set:
        X_H2 = (1.0 - np.sum(X_param, axis=0)) / (1.0 + He_fraction)
        X_He = He_fraction * X_H2
        X[0, :, :, :] = X_H2
        X[1, :, :, :] = X_He
    elif {"H", "He"} <= bulk_set and "H2" not in bulk_set:
        X_H = 2.0 * (1.0 - np.sum(X_param, axis=0)) / (1.0 + He_fraction)
        X_He = He_fraction * (X_H / 2.0)
        X[0, :, :, :] = X_H
        X[1, :, :, :] = X_He
    elif {"H2", "H", "He"} <= bulk_set:
        raise NotImplementedError(
            "H2/H/He dissociation bulk mixture deferred to v1 (POSEIDON "
            "atmosphere.py:1285-1312)."
        )
    else:
        if N_bulk_species > 1:
            raise Exception(
                "Only a single species can be designated as bulk (besides "
                "models with H2 & He or H & He with a fixed He/H2 ratio)."
            )
        X[0, :, :, :] = 1.0 - np.sum(X_param, axis=0)

    X[N_bulk_species:, :, :, :] = X_param
    return X


# ---------------------------------------------------------------------------
# Mean molecular mass
# ---------------------------------------------------------------------------
def compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all):
    """Mean molecular mass profile in kg, per POSEIDON.atmosphere.compute_mean_mol_mass
    (`POSEIDON/atmosphere.py:1816-1856`)."""
    N_layers = len(P)
    mu = np.zeros(shape=(N_layers, N_sectors, N_zones))
    for i in range(N_layers):
        for j in range(N_sectors):
            for k in range(N_zones):
                for q in range(N_species):
                    mu[i, j, k] += X[q, i, j, k] * masses_all[q]
    return mu * sc.u


# ---------------------------------------------------------------------------
# Radial profiles (hydrostatic)
# ---------------------------------------------------------------------------
def radial_profiles(P, T, g_0, R_p, P_ref, R_p_ref, mu, N_sectors, N_zones):
    """Inverse-square-gravity hydrostatic radius profile.

    Mirrors `POSEIDON/atmosphere.py:1494-1606` (analytic trapezoidal
    integral with 1/(1/r_0 + ∫H dlnP) form, NOT iterative).
    """
    N_layers = len(P)
    r = np.zeros((N_layers, N_sectors, N_zones))
    r_up = np.zeros((N_layers, N_sectors, N_zones))
    r_low = np.zeros((N_layers, N_sectors, N_zones))
    dr = np.zeros((N_layers, N_sectors, N_zones))
    n = np.zeros((N_layers, N_sectors, N_zones))
    log_P = np.log(P)

    for j in range(N_sectors):
        for k in range(N_zones):
            n[:, j, k] = (P * 1.0e5) / (sc.k * T[:, j, k])
            P_0 = P_ref
            r_0 = R_p_ref
            i_ref = int(np.argmin(np.abs(P - P_0)))
            r[i_ref, j, k] = r_0
            integrand = (sc.k * T[:, j, k]) / (R_p ** 2 * g_0 * mu[:, j, k])

            integral_out = 0.0
            integral_in = 0.0

            for i in range(i_ref + 1, N_layers):
                integral_out += 0.5 * (integrand[i] + integrand[i - 1]) * (log_P[i] - log_P[i - 1])
                r[i, j, k] = 1.0 / ((1.0 / r_0) + integral_out)

            for i in range(i_ref - 1, -1, -1):
                integral_in += 0.5 * (integrand[i] + integrand[i + 1]) * (log_P[i] - log_P[i + 1])
                r[i, j, k] = 1.0 / ((1.0 / r_0) + integral_in)

            for i in range(1, N_layers - 1):
                r_up[i, j, k] = 0.5 * (r[i + 1, j, k] + r[i, j, k])
                r_low[i, j, k] = 0.5 * (r[i, j, k] + r[i - 1, j, k])
                dr[i, j, k] = 0.5 * (r[i + 1, j, k] - r[i - 1, j, k])

            r_up[0, j, k] = 0.5 * (r[1, j, k] + r[0, j, k])
            r_up[N_layers - 1, j, k] = r[N_layers - 1, j, k] + 0.5 * (
                r[N_layers - 1, j, k] - r[N_layers - 2, j, k]
            )
            r_low[0, j, k] = r[0, j, k] - 0.5 * (r[1, j, k] - r[0, j, k])
            r_low[N_layers - 1, j, k] = 0.5 * (
                r[N_layers - 1, j, k] + r[N_layers - 2, j, k]
            )
            dr[0, j, k] = r[1, j, k] - r[0, j, k]
            dr[N_layers - 1, j, k] = (
                r[N_layers - 1, j, k] - r[N_layers - 2, j, k]
            )

    return n, r, r_up, r_low, dr


def radial_profiles_constant_g(P, T, g_0, P_ref, R_p_ref, mu, N_sectors, N_zones):
    """Constant-gravity hydrostatic radius profile.

    Mirrors `POSEIDON/atmosphere.py:1609-1722`. Used by
    test_TRIDENT.py::test_Rayleigh (`constant_gravity=True`).
    """
    N_layers = len(P)
    r = np.zeros((N_layers, N_sectors, N_zones))
    r_up = np.zeros((N_layers, N_sectors, N_zones))
    r_low = np.zeros((N_layers, N_sectors, N_zones))
    dr = np.zeros((N_layers, N_sectors, N_zones))
    n = np.zeros((N_layers, N_sectors, N_zones))
    log_P = np.log(P)

    for j in range(N_sectors):
        for k in range(N_zones):
            n[:, j, k] = (P * 1.0e5) / (sc.k * T[:, j, k])
            P_0 = P_ref
            r_0 = R_p_ref
            i_ref = int(np.argmin(np.abs(P - P_0)))
            r[i_ref, j, k] = r_0

            integral_out = 0.0
            integral_in = 0.0
            integrand = (sc.k * T[:, j, k]) / (g_0 * mu[:, j, k])

            for i in range(i_ref + 1, N_layers):
                integral_out += 0.5 * (integrand[i] + integrand[i - 1]) * (log_P[i] - log_P[i - 1])
                r[i, j, k] = r_0 - integral_out

            for i in range(i_ref - 1, -1, -1):
                integral_in += 0.5 * (integrand[i] + integrand[i + 1]) * (log_P[i] - log_P[i + 1])
                r[i, j, k] = r_0 - integral_in

            for i in range(1, N_layers - 1):
                r_up[i, j, k] = 0.5 * (r[i + 1, j, k] + r[i, j, k])
                r_low[i, j, k] = 0.5 * (r[i, j, k] + r[i - 1, j, k])
                dr[i, j, k] = 0.5 * (r[i + 1, j, k] - r[i - 1, j, k])

            r_up[0, j, k] = 0.5 * (r[1, j, k] + r[0, j, k])
            r_up[N_layers - 1, j, k] = r[N_layers - 1, j, k] + 0.5 * (
                r[N_layers - 1, j, k] - r[N_layers - 2, j, k]
            )
            r_low[0, j, k] = r[0, j, k] - 0.5 * (r[1, j, k] - r[0, j, k])
            r_low[N_layers - 1, j, k] = 0.5 * (
                r[N_layers - 1, j, k] + r[N_layers - 2, j, k]
            )
            dr[0, j, k] = r[1, j, k] - r[0, j, k]
            dr[N_layers - 1, j, k] = (
                r[N_layers - 1, j, k] - r[N_layers - 2, j, k]
            )

    return n, r, r_up, r_low, dr
