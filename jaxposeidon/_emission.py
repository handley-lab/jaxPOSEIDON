"""Thermal-emission + reflection forward model.

Ports POSEIDON `emission.py:30-1609`:
- planck_lambda_arr        — black-body spectral radiance
- emission_single_stream   — non-scattering thermal emission with Gauss quadrature
- emission_single_stream_w_albedo — variant with surface emissivity
- determine_photosphere_radii — τ → R(λ) interpolation
- slice_gt / setup_tri_diag / tri_diag_solve / numba_cumsum — Toon solver bits
- emission_Toon            — Toon source-function multiple-scattering
- reflection_Toon          — Toon multi-stream reflected light
- emission_bare_surface    — bare-rock thermal F = π · B · (1 - r_surf)
- reflection_bare_surface  — bare-rock reflected with 5-pt Gauss disk integral
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


def emission_single_stream_w_albedo(
    T,
    dz,
    wl,
    kappa,
    Gauss_quad=2,
    surf_reflect=None,
    index_below_P_surf=0,
):
    """Pure-thermal emission with hard-surface emissivity.

    Bit-equivalent port of POSEIDON `emission.py:181-261`.
    """
    if Gauss_quad == 2:
        W = np.array([0.5, 0.5])
        mu = np.array([0.5 - 0.5 * np.sqrt(1.0 / 3.0), 0.5 + 0.5 * np.sqrt(1.0 / 3.0)])
    elif Gauss_quad == 3:
        W = np.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
        mu = np.array(
            [0.5 - 0.5 * np.sqrt(3.0 / 5.0), 0.5, 0.5 + 0.5 * np.sqrt(3.0 / 5.0)]
        )

    if surf_reflect is None:
        surf_reflect = np.zeros(len(wl))
    emissivity = 1.0 - surf_reflect
    B = planck_lambda_arr(T, wl)
    B[index_below_P_surf, :] = B[index_below_P_surf, :] * emissivity

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


def determine_photosphere_radii(dtau, r_low, wl, photosphere_tau=2 / 3):
    """Wavelength-dependent photosphere radii via τ-interpolation.

    Bit-equivalent port of POSEIDON `emission.py:346-380`.
    """
    R_p_eff = np.zeros(len(wl))
    for k in range(len(wl)):
        tau_lambda = np.cumsum(dtau[:, k])
        R_p_eff[k] = np.interp(photosphere_tau, tau_lambda, r_low)
    return R_p_eff


def slice_gt(array, lim):
    """Clip per-row values above `lim` (POSEIDON `emission.py:423-432`)."""
    for i in range(array.shape[0]):
        new = array[i, :]
        new[np.where(new > lim)] = lim
        array[i, :] = new
    return array


def setup_tri_diag(
    N_layer,
    N_wl,
    c_plus_up,
    c_minus_up,
    c_plus_down,
    c_minus_down,
    b_top,
    b_surface,
    surf_reflect,
    gamma,
    dtau,
    exptrm_positive,
    exptrm_minus,
):
    """Build Toon+89 tridiagonal coefficients.

    Bit-equivalent port of POSEIDON `emission.py:436-530`.
    """
    L = 2 * N_layer
    e1 = exptrm_positive + gamma * exptrm_minus
    e2 = exptrm_positive - gamma * exptrm_minus
    e3 = gamma * exptrm_positive + exptrm_minus
    e4 = gamma * exptrm_positive - exptrm_minus

    A = np.zeros((L, N_wl))
    B = np.zeros((L, N_wl))
    C = np.zeros((L, N_wl))
    D = np.zeros((L, N_wl))

    A[0, :] = 0.0
    B[0, :] = gamma[0, :] + 1.0
    C[0, :] = gamma[0, :] - 1.0
    D[0, :] = b_top - c_minus_up[0, :]

    A[1::2, :][:-1] = (e1[:-1, :] + e3[:-1, :]) * (gamma[1:, :] - 1.0)
    B[1::2, :][:-1] = (e2[:-1, :] + e4[:-1, :]) * (gamma[1:, :] - 1.0)
    C[1::2, :][:-1] = 2.0 * (1.0 - gamma[1:, :] ** 2)
    D[1::2, :][:-1] = (gamma[1:, :] - 1.0) * (
        c_plus_up[1:, :] - c_plus_down[:-1, :]
    ) + (1.0 - gamma[1:, :]) * (c_minus_down[:-1, :] - c_minus_up[1:, :])

    A[::2, :][1:] = 2.0 * (1.0 - gamma[:-1, :] ** 2)
    B[::2, :][1:] = (e1[:-1, :] - e3[:-1, :]) * (gamma[1:, :] + 1.0)
    C[::2, :][1:] = (e1[:-1, :] + e3[:-1, :]) * (gamma[1:, :] - 1.0)
    D[::2, :][1:] = e3[:-1, :] * (c_plus_up[1:, :] - c_plus_down[:-1, :]) + e1[
        :-1, :
    ] * (c_minus_down[:-1, :] - c_minus_up[1:, :])

    A[-1, :] = e1[-1, :] - surf_reflect * e3[-1, :]
    B[-1, :] = e2[-1, :] - surf_reflect * e4[-1, :]
    C[-1, :] = 0.0
    D[-1, :] = b_surface - c_plus_down[-1, :] + surf_reflect * c_minus_down[-1, :]

    return A, B, C, D


def tri_diag_solve(l, a, b, c, d):
    """Tridiagonal Thomas-algorithm solve A·X = D for a single wavelength.

    Bit-equivalent port of POSEIDON `emission.py:534-569`.
    """
    AS = np.zeros(l)
    DS = np.zeros(l)
    XK = np.zeros(l)
    AS[-1] = a[-1] / b[-1]
    DS[-1] = d[-1] / b[-1]
    for i in range(l - 2, -1, -1):
        x = 1.0 / (b[i] - c[i] * AS[i + 1])
        AS[i] = a[i] * x
        DS[i] = (d[i] - c[i] * DS[i + 1]) * x
    XK[0] = DS[0]
    for i in range(1, l):
        XK[i] = DS[i] - AS[i] * XK[i - 1]
    return XK


def numba_cumsum(mat):
    """Column-wise cumulative sum.

    Bit-equivalent port of POSEIDON `emission.py:966-973`.
    """
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:, i] = np.cumsum(mat[:, i])
    return new_mat


def emission_Toon(
    P,
    T,
    wl,
    dtau_tot,
    kappa_Ray,
    kappa_cloud,
    kappa_tot,
    w_cloud,
    g_cloud,
    zone_idx,
    surf_reflect,
    kappa_cloud_seperate,
    hard_surface=0,
    tridiagonal=0,
    Gauss_quad=5,
    numt=1,
    T_surf=0,
):
    """Toon two-stream multiple-scattering thermal emission.

    Bit-equivalent port of POSEIDON `emission.py:573-963`. PICASO source-
    function method with Toon+89 quadrature. Hemispheric mean (μ₁=0.5).
    """
    kappa_cloud_w_cloud_sum = np.zeros_like(kappa_cloud)
    kappa_cloud_w_cloud_g_cloud_sum = np.zeros_like(kappa_cloud)

    for aerosol in range(len(kappa_cloud_seperate)):
        w_cloud[aerosol, :, 0, zone_idx, :] = (
            w_cloud[aerosol, :, 0, zone_idx, :] * 0.99999
        )
        kappa_cloud_w_cloud_sum[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            * w_cloud[aerosol, :, 0, zone_idx, :]
        )
        kappa_cloud_w_cloud_g_cloud_sum[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            * w_cloud[aerosol, :, 0, zone_idx, :]
            * g_cloud[aerosol, :, 0, zone_idx, :]
        )

    w_tot = (
        (0.99999 * kappa_Ray[:, 0, zone_idx, :])
        + (kappa_cloud_w_cloud_sum[:, 0, zone_idx, :])
    ) / kappa_tot
    g_tot = (kappa_cloud_w_cloud_g_cloud_sum[:, 0, zone_idx, :]) / (
        kappa_cloud_w_cloud_sum[:, 0, zone_idx, :]
        + (0.99999 * kappa_Ray[:, 0, zone_idx, :])
    )

    P = np.flipud(P)
    T = np.flipud(T)
    dtau_tot = np.flipud(dtau_tot)
    w_tot = np.flipud(w_tot)
    g_tot = np.flipud(g_tot)

    N_wl = len(wl)
    N_layer = len(P)
    N_level = N_layer + 1

    T_level = np.zeros(N_level)
    log_P_level = np.zeros(N_level)
    T_level[1:-1] = (T[1:] + T[:-1]) / 2.0
    T_level[0] = T_level[1] - (T_level[2] - T_level[1])
    T_level[-1] = T_level[-2] + (T_level[-2] - T_level[-3])

    log_P = np.log10(P)
    log_P_level[1:-1] = (log_P[1:] + log_P[:-1]) / 2.0
    log_P_level[0] = log_P_level[1] - (log_P_level[2] - log_P_level[1])
    log_P_level[-1] = log_P_level[-2] + (log_P_level[-2] - log_P_level[-3])
    P_level = np.power(10.0, log_P_level)

    gangle = np.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = np.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )

    cos_theta = 1.0
    longitude = np.arcsin(
        (gangle - (cos_theta - 1.0) / (cos_theta + 1.0)) / (2.0 / (cos_theta + 1))
    )
    colatitude = np.arccos(0.0)
    f = np.sin(colatitude)
    ubar1 = np.outer(np.cos(longitude), f)  # noqa: F841

    mu1 = 0.5
    all_b = planck_lambda_arr(T_level, wl)
    b0 = all_b[0:-1, :]
    b1 = (all_b[1:, :] - b0) / dtau_tot

    g1 = 2.0 - (w_tot * (1 + g_tot))
    g2 = w_tot * (1 - g_tot)
    alpha = np.sqrt((1.0 - w_tot) / (1.0 - (w_tot * g_tot)))  # noqa: F841
    lamda = np.sqrt(g1**2 - g2**2)
    gamma = (g1 - lamda) / g2
    g1_plus_g2 = 1.0 / (g1 + g2)

    c_plus_up = 2 * np.pi * mu1 * (b0 + b1 * g1_plus_g2)
    c_minus_up = 2 * np.pi * mu1 * (b0 - b1 * g1_plus_g2)
    c_plus_down = 2 * np.pi * mu1 * (b0 + b1 * dtau_tot + b1 * g1_plus_g2)
    c_minus_down = 2 * np.pi * mu1 * (b0 + b1 * dtau_tot - b1 * g1_plus_g2)

    exptrm = lamda * dtau_tot
    exptrm = slice_gt(exptrm, 35.0)
    exptrm_positive = np.exp(exptrm)
    exptrm_minus = 1.0 / exptrm_positive

    tau_top = dtau_tot[0, :] * P_level[0] / (P_level[1] - P_level[0])
    b_top = (1.0 - np.exp(-tau_top / mu1)) * all_b[0, :] * np.pi
    if hard_surface:
        emissivity = 1.0 - surf_reflect
        b_surface = emissivity * all_b[-1, :] * np.pi
    else:
        b_surface = (all_b[-1, :] + b1[-1, :] * mu1) * np.pi

    A, B, C, D = setup_tri_diag(
        N_layer,
        N_wl,
        c_plus_up,
        c_minus_up,
        c_plus_down,
        c_minus_down,
        b_top,
        b_surface,
        surf_reflect,
        gamma,
        dtau_tot,
        exptrm_positive,
        exptrm_minus,
    )
    positive = np.zeros((N_layer, N_wl))
    negative = np.zeros((N_layer, N_wl))
    L = N_layer + N_layer
    for w in range(N_wl):
        X = tri_diag_solve(L, A[:, w], B[:, w], C[:, w], D[:, w])
        positive[:, w] = X[::2] + X[1::2]
        negative[:, w] = X[::2] - X[1::2]

    f_up = positive * exptrm_positive + gamma * negative * exptrm_minus + c_plus_up  # noqa: F841

    G = (1 / mu1 - lamda) * positive
    H = gamma * (lamda + 1 / mu1) * negative
    J = gamma * (lamda + 1 / mu1) * positive  # noqa: F841
    K = (1 / mu1 - lamda) * negative  # noqa: F841
    alpha1 = 2 * np.pi * (b0 + b1 * (g1_plus_g2 - mu1))
    alpha2 = 2 * np.pi * b1
    sigma1 = 2 * np.pi * (b0 - b1 * (g1_plus_g2 - mu1))  # noqa: F841
    sigma2 = 2 * np.pi * b1  # noqa: F841

    int_minus = np.zeros((N_level, N_wl))
    int_plus = np.zeros((N_level, N_wl))
    int_minus_mdpt = np.zeros((N_level, N_wl))  # noqa: F841
    int_plus_mdpt = np.zeros((N_level, N_wl))

    exptrm_positive_mdpt = np.exp(0.5 * exptrm)
    exptrm_minus_mdpt = 1 / exptrm_positive_mdpt

    int_at_top = np.zeros((Gauss_quad, numt, N_wl))
    int_down = np.zeros((Gauss_quad, numt, N_wl))  # noqa: F841

    F = np.zeros(N_wl)

    for ng in range(Gauss_quad):
        for nt in range(numt):
            iubar = gangle[ng]
            if hard_surface:
                emissivity = 1.0 - surf_reflect
                int_plus[-1, :] = emissivity * all_b[-1, :] * 2 * np.pi
            else:
                int_plus[-1, :] = (all_b[-1, :] + b1[-1, :] * iubar) * 2 * np.pi

            int_minus[0, :] = (1 - np.exp(-tau_top / iubar)) * all_b[0, :] * 2 * np.pi

            exptrm_angle = np.exp(-dtau_tot / iubar)
            exptrm_angle_mdpt = np.exp(-0.5 * dtau_tot / iubar)

            for itop in range(N_layer):
                ibot = N_layer - 1 - itop
                int_plus[ibot, :] = (
                    int_plus[ibot + 1, :] * exptrm_angle[ibot, :]
                    + (G[ibot, :] / (lamda[ibot, :] * iubar - 1.0))
                    * (exptrm_positive[ibot, :] * exptrm_angle[ibot, :] - 1.0)
                    + (H[ibot, :] / (lamda[ibot, :] * iubar + 1.0))
                    * (1.0 - exptrm_minus[ibot, :] * exptrm_angle[ibot, :])
                    + alpha1[ibot, :] * (1.0 - exptrm_angle[ibot, :])
                    + alpha2[ibot, :]
                    * (iubar - (dtau_tot[ibot, :] + iubar) * exptrm_angle[ibot, :])
                )

                int_plus_mdpt[ibot, :] = (
                    int_plus[ibot + 1, :] * exptrm_angle_mdpt[ibot, :]
                    + (G[ibot, :] / (lamda[ibot, :] * iubar - 1.0))
                    * (
                        exptrm_positive[ibot, :] * exptrm_angle_mdpt[ibot, :]
                        - exptrm_positive_mdpt[ibot, :]
                    )
                    - (H[ibot, :] / (lamda[ibot, :] * iubar + 1.0))
                    * (
                        exptrm_minus[ibot, :] * exptrm_angle_mdpt[ibot, :]
                        - exptrm_minus_mdpt[ibot, :]
                    )
                    + alpha1[ibot, :] * (1.0 - exptrm_angle_mdpt[ibot, :])
                    + alpha2[ibot, :]
                    * (
                        iubar
                        + 0.5 * dtau_tot[ibot, :]
                        - (dtau_tot[ibot, :] + iubar) * exptrm_angle_mdpt[ibot, :]
                    )
                )

            int_at_top[ng, nt, :] = int_plus_mdpt[0, :]

    for ng in range(Gauss_quad):
        F += int_at_top[ng, 0, :] * gweight[ng]

    return F, dtau_tot


def reflection_Toon(
    P,
    wl,
    dtau_tot,
    kappa_Ray,
    kappa_cloud,
    kappa_tot,
    w_cloud,
    g_cloud,
    zone_idx,
    surf_reflect,
    kappa_cloud_seperate,
    single_phase=3,
    multi_phase=0,
    frac_a=1,
    frac_b=-1,
    frac_c=2,
    constant_back=-0.5,
    constant_forward=1,
    Gauss_quad=5,
    numt=1,
    toon_coefficients=0,
    tridiagonal=0,
    b_top=0,
):
    """Toon multi-stream reflected-light flux.

    Bit-equivalent port of POSEIDON `emission.py:976-1573`. TTHG+Rayleigh
    single-scattering phase function with quadrature Toon coefficients.

    Note: `single_phase` is exposed for API parity with POSEIDON but is
    inert — POSEIDON only retains the `single_phase==3` (TTHG_ray) branch
    active; the other three branches (`0`/`'cahoy'`, `1`/`'OTHG'`,
    `2`/`'TTHG'`) are commented out in POSEIDON's source.
    """
    N_wl = len(wl)
    N_layer = len(P)
    N_level = N_layer + 1  # noqa: F841

    kappa_cloud_w_cloud_sum = np.zeros_like(kappa_cloud)
    kappa_cloud_g_cloud_sum = np.zeros_like(kappa_cloud)
    kappa_cloud_w_cloud_g_cloud_sum = np.zeros_like(kappa_cloud)
    g_cloud_tot_weighted = np.zeros_like(kappa_cloud)

    for aerosol in range(len(kappa_cloud_seperate)):
        w_cloud[aerosol, :, 0, zone_idx, :] = (
            w_cloud[aerosol, :, 0, zone_idx, :] * 0.99999
        )
        kappa_cloud_w_cloud_sum[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            * w_cloud[aerosol, :, 0, zone_idx, :]
        )
        kappa_cloud_g_cloud_sum[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            * g_cloud[aerosol, :, 0, zone_idx, :]
        )
        kappa_cloud_w_cloud_g_cloud_sum[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            * w_cloud[aerosol, :, 0, zone_idx, :]
            * g_cloud[aerosol, :, 0, zone_idx, :]
        )
        g_cloud_tot_weighted[:, 0, zone_idx, :] += (
            kappa_cloud_seperate[aerosol, :, 0, zone_idx, :]
            / kappa_cloud[:, 0, zone_idx, :]
        ) * g_cloud[aerosol, :, 0, zone_idx, :]

    np.nan_to_num(g_cloud_tot_weighted, copy=False, nan=0.0)

    ftau_cld = (kappa_cloud_w_cloud_sum[:, 0, zone_idx, :]) / (
        kappa_cloud_w_cloud_sum[:, 0, zone_idx, :]
        + (0.99999 * kappa_Ray[:, 0, zone_idx, :])
    )
    ftau_ray = kappa_Ray[:, 0, zone_idx, :] / (
        kappa_Ray[:, 0, zone_idx, :] + kappa_cloud_g_cloud_sum[:, 0, zone_idx, :]
    )
    gcos2 = 0.5 * ftau_ray
    w_tot = (
        (0.99999 * kappa_Ray[:, 0, zone_idx, :])
        + (kappa_cloud_w_cloud_sum[:, 0, zone_idx, :])
    ) / kappa_tot

    P = np.flipud(P)
    dtau_tot = np.flipud(dtau_tot)
    w_tot = np.flipud(w_tot)
    g_cloud_tot_weighted = np.flipud(g_cloud_tot_weighted)
    ftau_cld = np.flipud(ftau_cld)
    ftau_ray = np.flipud(ftau_ray)
    g_cloud_tot_weighted = g_cloud_tot_weighted[:, 0, zone_idx, :]

    tau = np.zeros((N_layer + 1, N_wl))
    tau[1:, :] = numba_cumsum(dtau_tot[:, :])

    stream = 2
    f_deltaM = g_cloud_tot_weighted**stream
    w_dedd = w_tot * (1.0 - f_deltaM) / (1.0 - w_tot * f_deltaM)
    g_dedd = (g_cloud_tot_weighted - f_deltaM) / (1.0 - f_deltaM)
    dtau_dedd = dtau_tot * (1.0 - w_tot * f_deltaM)
    tau_dedd = np.zeros((N_layer + 1, N_wl))
    tau_dedd[1:, :] = numba_cumsum(dtau_dedd[:, :])

    gangle = np.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = np.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )

    phase_angle = 0
    cos_theta = 1.0
    longitude = np.arcsin(
        (gangle - (cos_theta - 1.0) / (cos_theta + 1.0)) / (2.0 / (cos_theta + 1))
    )
    colatitude = np.arccos(0.0)
    f = np.sin(colatitude)
    ubar0 = np.outer(np.cos(longitude - phase_angle), f)
    ubar1 = np.outer(np.cos(longitude), f)
    F0PI = np.zeros(N_wl) + 1

    xint_at_top = np.zeros((Gauss_quad, numt, N_wl))

    sq3 = np.sqrt(3.0)
    if toon_coefficients == 1:
        g1 = (7 - w_dedd * (4 + 3 * ftau_cld * g_dedd)) / 4
        g2 = -(1 - w_dedd * (4 - 3 * ftau_cld * g_dedd)) / 4
    else:
        g1 = (sq3 * 0.5) * (2.0 - w_dedd * (1.0 + ftau_cld * g_dedd))
        g2 = (sq3 * w_dedd * 0.5) * (1.0 - ftau_cld * g_dedd)

    lamda = np.sqrt(g1**2 - g2**2)
    gama = (g1 - lamda) / g2

    for ng in range(Gauss_quad):
        for nt in range(numt):
            u1 = ubar1[ng, nt]
            u0 = ubar0[ng, nt]
            if toon_coefficients == 1:
                g3 = (2 - 3 * ftau_cld * g_dedd * u0) / 4
            else:
                g3 = 0.5 * (1.0 - sq3 * ftau_cld * g_dedd * u0)

            a_minus = (
                F0PI
                * w_dedd
                * ((1.0 - g3) * (g1 + 1.0 / u0) + g2 * g3)
                / (lamda**2 - 1.0 / u0**2.0)
            )
            a_plus = (
                F0PI
                * w_dedd
                * (g3 * (g1 - 1.0 / u0) + g2 * (1.0 - g3))
                / (lamda**2 - 1.0 / u0**2.0)
            )

            x = np.exp(-tau_dedd[:-1, :] / u0)
            c_minus_up = a_minus * x
            c_plus_up = a_plus * x
            x = np.exp(-tau_dedd[1:, :] / u0)
            c_minus_down = a_minus * x
            c_plus_down = a_plus * x

            exptrm = lamda * dtau_dedd
            exptrm = slice_gt(exptrm, 35.0)
            exptrm_positive = np.exp(exptrm)
            exptrm_minus = 1.0 / exptrm_positive

            b_surface = 0.0 + surf_reflect * u0 * F0PI * np.exp(-tau_dedd[-1, :] / u0)

            A, B, C, D = setup_tri_diag(
                N_layer,
                N_wl,
                c_plus_up,
                c_minus_up,
                c_plus_down,
                c_minus_down,
                b_top,
                b_surface,
                surf_reflect,
                gama,
                dtau_dedd,
                exptrm_positive,
                exptrm_minus,
            )

            positive = np.zeros((N_layer, N_wl))
            negative = np.zeros((N_layer, N_wl))
            L = N_layer + N_layer
            for w in range(N_wl):
                X = tri_diag_solve(L, A[:, w], B[:, w], C[:, w], D[:, w])
                positive[:, w] = X[::2] + X[1::2]
                negative[:, w] = X[::2] - X[1::2]

            xint = np.zeros((N_layer + 1, N_wl))
            xint[-1, :] = (
                positive[-1, :] * exptrm_positive[-1, :]
                + gama[-1, :] * negative[-1, :] * exptrm_minus[-1, :]
                + c_plus_down[-1, :]
            ) / np.pi

            if multi_phase == 0:
                ubar2 = 0.767
                multi_plus = (
                    1.0
                    + 1.5 * ftau_cld * g_dedd * u1
                    + gcos2 * (3.0 * ubar2 * ubar2 * u1 * u1 - 1.0) / 2.0
                )
                multi_minus = (
                    1.0
                    - 1.5 * ftau_cld * g_dedd * u1
                    + gcos2 * (3.0 * ubar2 * ubar2 * u1 * u1 - 1.0) / 2.0
                )
            elif multi_phase == 1:
                multi_plus = 1.0 + 1.5 * ftau_cld * g_dedd * u1
                multi_minus = 1.0 - 1.5 * ftau_cld * g_dedd * u1

            G = positive * (multi_plus + gama * multi_minus) * w_dedd
            H = negative * (gama * multi_plus + multi_minus) * w_dedd
            A_ms = (multi_plus * c_plus_up + multi_minus * c_minus_up) * w_dedd
            G *= 0.5 / np.pi
            H *= 0.5 / np.pi
            A_ms *= 0.5 / np.pi

            g_forward = constant_forward * g_cloud_tot_weighted
            g_back = constant_back * g_cloud_tot_weighted

            p_single = ftau_cld * (
                (frac_a + frac_b * g_back**frac_c)
                * (1 - g_forward**2)
                / np.sqrt((1 + g_forward**2 + 2 * g_forward * cos_theta) ** 3)
                + (1 - (frac_a + frac_b * g_back**frac_c))
                * (1 - g_back**2)
                / np.sqrt((1 + g_back**2 + 2 * g_back * cos_theta) ** 3)
            ) + ftau_ray * (0.75 * (1 + cos_theta**2.0))

            for i in range(N_layer - 1, -1, -1):
                xint[i, :] = (
                    xint[i + 1, :] * np.exp(-dtau_dedd[i, :] / u1)
                    + (w_tot[i, :] * F0PI / (4.0 * np.pi))
                    * (p_single[i, :])
                    * np.exp(-tau[i, :] / u0)
                    * (1.0 - np.exp(-dtau_tot[i, :] * (u0 + u1) / (u0 * u1)))
                    * (u0 / (u0 + u1))
                    + A_ms[i, :]
                    * (1.0 - np.exp(-dtau_dedd[i, :] * (u0 + 1 * u1) / (u0 * u1)))
                    * (u0 / (u0 + 1 * u1))
                    + G[i, :]
                    * (np.exp(exptrm[i, :] * 1 - dtau_dedd[i, :] / u1) - 1.0)
                    / (lamda[i, :] * 1 * u1 - 1.0)
                    + H[i, :]
                    * (1.0 - np.exp(-exptrm[i, :] * 1 - dtau_dedd[i, :] / u1))
                    / (lamda[i, :] * 1 * u1 + 1.0)
                )

            xint_at_top[ng, nt, :] = xint[0, :]

    tweight = np.array([1])
    sym_fac = 2 * np.pi
    albedo = np.zeros(N_wl)
    for ig in range(len(gweight)):
        for it in range(len(tweight)):
            albedo = albedo + xint_at_top[ig, it, :] * gweight[ig] * tweight[it]
    albedo = sym_fac * 0.5 * albedo / F0PI * (cos_theta + 1.0)
    return albedo


def build_surf_reflect(
    wl,
    surface,
    surface_model,
    albedo_deck,
    albedo_surf,
    surface_components,
    surface_component_albedos,
    surface_component_percentages,
    surface_percentage_apply_to,
):
    """Construct (surf_reflect, surf_reflect_array) per POSEIDON `core.py:1527-1556`/`1741-1770`."""
    from jaxposeidon._surface_setup import interpolate_surface_components

    if surface or albedo_deck != -1:
        if surface:
            if surface_model == "gray":
                surf_reflect = np.zeros_like(wl)
                surf_reflect_array = []
            elif surface_model == "constant":
                surf_reflect = np.full_like(wl, albedo_surf)
                surf_reflect_array = []
            elif surface_model == "lab_data":
                surf_reflect_array = interpolate_surface_components(
                    wl, surface_components, surface_component_albedos
                )
                if surface_percentage_apply_to == "albedos":
                    surf_reflect = np.zeros_like(wl)
                    for n in range(len(surface_component_percentages)):
                        surf_reflect += (
                            surface_component_percentages[n] * surf_reflect_array[n]
                        )
                else:
                    surf_reflect = np.full_like(wl, -1.0)
        else:
            surf_reflect = np.full_like(wl, albedo_deck)
            surf_reflect_array = []
    else:
        surf_reflect = np.full_like(wl, -1.0)
        surf_reflect_array = []
    return surf_reflect, surf_reflect_array


def assign_assumptions_and_compute_single_stream_emission(
    P,
    T,
    dz,
    wl,
    kappa_tot,
    dtau_tot,
    kappa_gas,
    kappa_Ray,
    kappa_cloud,
    kappa_cloud_seperate,
    zone_idx,
    Gauss_quad,
    P_cloud,
    cloud_dim,
    aerosol_species,
    f_cloud,
    albedo_deck,
    disable_atmosphere,
    surface,
    surface_model,
    P_surf,
    T_surf,
    surf_reflect,
    surf_reflect_array,
    surface_component_percentages,
    surface_percentage_apply_to,
):
    """Surface/albedo-aware single-stream emission orchestrator.

    Bit-equivalent port of POSEIDON `emission.py:1681-1878` (CPU-only
    subset; thermal_scattering branch and GPU exits are not ported).
    """
    from jaxposeidon._surface_setup import find_nearest_less_than

    if cloud_dim == 2:
        kappa_cloud_clear = np.zeros_like(kappa_cloud)
        kappa_tot_clear = (
            kappa_gas[:, 0, zone_idx, :]
            + kappa_Ray[:, 0, zone_idx, :]
            + kappa_cloud_clear[:, 0, zone_idx, :]
        )
        dtau_tot_clear = np.ascontiguousarray(  # noqa: F841
            kappa_tot_clear * dz.reshape((len(P), 1))
        )
        if len(aerosol_species) >= 2:
            raise Exception(
                "In single stream emission, only one aerosol species can be "
                "patchy. For two, use thermal_scattering = True."
            )

    if surface or albedo_deck != -1:
        if not disable_atmosphere:
            if surface:
                index_below_P_surf = find_nearest_less_than(P_surf, P)
                if index_below_P_surf + 1 != len(P):
                    index_below_P_surf -= 1
            else:
                try:
                    index_below_P_surf = find_nearest_less_than(P_cloud, P)
                except Exception:
                    index_below_P_surf = find_nearest_less_than(P_cloud[0], P)
                if index_below_P_surf + 1 != len(P):
                    index_below_P_surf -= 1

        if (
            (surface_model == "gray")
            or (surface_model == "constant")
            or (albedo_deck != -1)
            or (
                surface_model == "lab_data" and surface_percentage_apply_to == "albedos"
            )
        ):
            if not disable_atmosphere:
                F_p, _dtau = emission_single_stream_w_albedo(
                    T, dz, wl, kappa_tot, Gauss_quad, surf_reflect, index_below_P_surf
                )
                if cloud_dim == 2:
                    F_p_clear, _ = emission_single_stream_w_albedo(
                        T,
                        dz,
                        wl,
                        kappa_tot_clear,
                        Gauss_quad,
                        surf_reflect,
                        index_below_P_surf,
                    )
                    F_p = (f_cloud * F_p) + ((1 - f_cloud) * F_p_clear)
            else:
                F_p = emission_bare_surface(T_surf, wl, surf_reflect)
        elif surface_model == "lab_data" and surface_percentage_apply_to == "models":
            F_p_array = []
            for surf_reflect_n in surf_reflect_array:
                if not disable_atmosphere:
                    F_p_temp, _ = emission_single_stream_w_albedo(
                        T,
                        dz,
                        wl,
                        kappa_tot,
                        Gauss_quad,
                        surf_reflect_n,
                        index_below_P_surf,
                    )
                    if cloud_dim == 2:
                        F_p_clear, _ = emission_single_stream_w_albedo(
                            T,
                            dz,
                            wl,
                            kappa_tot_clear,
                            Gauss_quad,
                            surf_reflect_n,
                            index_below_P_surf,
                        )
                        F_p_temp = (f_cloud * F_p_temp) + ((1 - f_cloud) * F_p_clear)
                else:
                    F_p_temp = emission_bare_surface(T_surf, wl, surf_reflect_n)
                F_p_array.append(F_p_temp)
            F_p = np.zeros_like(wl)
            for n in range(len(surface_component_percentages)):
                F_p += surface_component_percentages[n] * F_p_array[n]

        if not disable_atmosphere:
            dtau = dtau_tot
        else:
            dtau = 0
    else:
        F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
        dtau = np.flip(dtau, axis=0)
        if cloud_dim == 2:
            F_p_clear, dtau = emission_single_stream(
                T, dz, wl, kappa_tot_clear, Gauss_quad
            )
            F_p = (f_cloud * F_p) + ((1 - f_cloud) * F_p_clear)

    return F_p, dtau


def reflection_bare_surface(wl, surf_reflect, Gauss_quad=5):
    """Bare-rock reflected-light albedo with 5-pt Gaussian disk integration.

    Bit-equivalent port of POSEIDON `emission.py:1612-1700` simplified
    (assumes phase_angle=0, F0PI=1).
    """
    N_wl = len(wl)
    gangle = np.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = np.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )
    tangle = np.array([0])  # noqa: F841
    tweight = np.array([1])
    cos_theta = 1.0
    F0PI = np.zeros(N_wl) + 1
    phase_angle = 0
    longitude = np.arcsin(
        (gangle - (cos_theta - 1.0) / (cos_theta + 1.0)) / (2.0 / (cos_theta + 1))
    )
    colatitude = np.arccos(0.0)
    f = np.sin(colatitude)
    ubar0 = np.outer(np.cos(longitude - phase_angle), f)
    ubar1 = np.outer(np.cos(longitude), f)  # noqa: F841

    sym_fac = 2 * np.pi
    xint_at_top = np.zeros((len(gangle), len(tweight), N_wl))
    for ig in range(len(gangle)):
        for _it in range(len(tweight)):
            u0 = ubar0[ig, 0]
            xint_at_top[ig, 0, :] = surf_reflect * u0 * F0PI / np.pi

    albedo = np.zeros(N_wl)
    for ig in range(len(gweight)):
        for it in range(len(tweight)):
            albedo += xint_at_top[ig, it, :] * gweight[ig] * tweight[it]
    albedo = sym_fac * 0.5 * albedo / F0PI * (cos_theta + 1.0)
    return albedo
