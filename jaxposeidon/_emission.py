"""Thermal-emission forward model.

Ports POSEIDON `emission.py:30-178` and `:1576-1609`:
- planck_lambda_arr        — black-body spectral radiance
- emission_single_stream   — non-scattering thermal emission with Gauss quadrature
- emission_bare_surface    — bare-rock emission (pure Planck × emissivity × π)

Full Toon two-stream (`emission_Toon`, `reflection_Toon`, ~1500 lines)
is the follow-up phase 0.5.13b — needs scattering, multi-stream radiative
transfer with banded matrix solves.
"""

import numpy as np
import scipy.constants as sc


def planck_lambda_arr(T, wl):
    """Black-body spectral radiance B(T, λ) in SI (W/m²/sr/m).

    Bit-equivalent port of POSEIDON `emission.py:30-69`.
    """
    B_lambda = np.zeros(shape=(len(T), len(wl)))
    wl_m = wl * 1.0e-6
    c_2 = (sc.h * sc.c) / sc.k
    for k in range(len(wl)):
        coeff = (2.0 * sc.h * sc.c**2) / (wl_m[k] ** 5)
        for i in range(len(T)):
            B_lambda[i, k] = coeff * (1.0 / (np.exp(c_2 / (wl_m[k] * T[i])) - 1.0))
    return B_lambda


def emission_single_stream(T, dz, wl, kappa, Gauss_quad=2):
    """Pure thermal emission with no scattering (Gauss-quadrature solver).

    Bit-equivalent port of POSEIDON `emission.py:111-178`. Returns
    `(F, dtau)` where `F` is the top-of-atmosphere surface flux (W/m²/sr/m)
    and `dtau` is the per-layer vertical optical depth.
    """
    if Gauss_quad == 2:
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5 * np.sqrt(1.0 / 3.0), 0.5 + 0.5 * np.sqrt(1.0 / 3.0)])
    elif Gauss_quad == 3:
        W = np.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
        mu = np.array(
            [0.5 - 0.5 * np.sqrt(3.0 / 5.0), 0.5, 0.5 + 0.5 * np.sqrt(3.0 / 5.0)]
        )
    else:
        raise NotImplementedError(f"Gauss_quad={Gauss_quad} not in {{2, 3}}")

    B = planck_lambda_arr(T, wl)
    I = np.ones(shape=(len(mu), len(wl))) * B[0, :]
    F = np.zeros(len(wl))
    dtau = np.zeros(shape=(len(T), len(wl)))

    for k in range(len(wl)):
        for j in range(len(mu)):
            for i in range(len(T)):
                dtau_vert = kappa[i, k] * dz[i]
                dtau[i, k] = dtau_vert
                Trans = np.exp((-1.0 * dtau_vert) / mu[j])
                I[j, k] = Trans * I[j, k] + (1.0 - Trans) * B[i, k]
            F[k] += 2.0 * np.pi * mu[j] * I[j, k] * W[j]

    return F, dtau


def emission_bare_surface(T_surf, wl, surf_reflect):
    """Bare-rock emergent thermal flux: F = π · B(T_surf, λ) · (1 - surf_reflect).

    Bit-equivalent port of POSEIDON `emission.py:1576-1609`.
    """
    T = np.array([T_surf])
    B = planck_lambda_arr(T, wl)
    emissivity = 1.0 - surf_reflect
    return B[0, :] * emissivity * np.pi
