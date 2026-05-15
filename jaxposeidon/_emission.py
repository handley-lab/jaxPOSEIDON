"""Thermal-emission + reflection forward model (v1-D JAX port).

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

v1-D conversion: numpy operations replaced with `jnp` so the hot path is
jax-traceable. Thomas tridiagonal sweep uses `lax.scan`. Python-level
shape constants (N_layer, N_wl, Gauss_quad) are treated as static under
`jit`.
"""

import jax
import jax.numpy as jnp
from jax import lax

from jaxposeidon._constants import BOLTZMANN_K, C_M_PER_S, PLANCK_H

jax.config.update("jax_enable_x64", True)

_H = PLANCK_H
_C = C_M_PER_S
_K = BOLTZMANN_K


def planck_lambda_arr(T, wl):
    """Black-body spectral radiance B(T, λ) in SI (W/m²/sr/m).

    Bit-equivalent port of POSEIDON `emission.py:30-69`. Returns a
    (len(T), len(wl)) array.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    wl_m = wl * 1.0e-6
    c_2 = (_H * _C) / _K
    coeff = (2.0 * _H * _C**2) / (wl_m**5)
    denom = jnp.exp(c_2 / (wl_m[None, :] * T[:, None])) - 1.0
    return coeff[None, :] / denom


def emission_single_stream(T, dz, wl, kappa, Gauss_quad=2):
    """Pure thermal emission with no scattering (Gauss-quadrature solver).

    Bit-equivalent port of POSEIDON `emission.py:111-178`. Returns
    `(F, dtau)`.
    """
    W, mu = _gauss_quad(Gauss_quad)
    return _emission_single_stream_inner(T, dz, wl, kappa, W, mu)


def _gauss_quad(Gauss_quad):
    if Gauss_quad == 2:
        W = jnp.array([0.5, 0.5])
        mu = jnp.array(
            [0.5 - 0.5 * jnp.sqrt(1.0 / 3.0), 0.5 + 0.5 * jnp.sqrt(1.0 / 3.0)]
        )
    elif Gauss_quad == 3:
        W = jnp.array([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
        mu = jnp.array(
            [
                0.5 - 0.5 * jnp.sqrt(3.0 / 5.0),
                0.5,
                0.5 + 0.5 * jnp.sqrt(3.0 / 5.0),
            ]
        )
    return W, mu


def _emission_single_stream_inner(T, dz, wl, kappa, W, mu):
    T = jnp.asarray(T, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)

    B = planck_lambda_arr(T, wl)
    N_T = T.shape[0]
    N_wl = wl.shape[0]
    N_mu = mu.shape[0]

    dtau = kappa * dz[:, None]
    I0 = jnp.broadcast_to(B[0, :], (N_mu, N_wl))

    def scan_layer(I, i):
        Trans = jnp.exp(-dtau[i, :][None, :] / mu[:, None])
        I_new = Trans * I + (1.0 - Trans) * B[i, :][None, :]
        return I_new, None

    I_final, _ = lax.scan(scan_layer, I0, jnp.arange(N_T))
    F = jnp.sum(2.0 * jnp.pi * mu[:, None] * I_final * W[:, None], axis=0)
    return F, dtau


def emission_bare_surface(T_surf, wl, surf_reflect):
    """Bare-rock emergent thermal flux: F = π · B(T_surf, λ) · (1 - surf_reflect).

    Bit-equivalent port of POSEIDON `emission.py:1576-1609`.
    """
    T_surf = jnp.asarray(T_surf, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    surf_reflect = jnp.asarray(surf_reflect, dtype=jnp.float64)
    T = jnp.atleast_1d(T_surf)
    B = planck_lambda_arr(T, wl)
    emissivity = 1.0 - surf_reflect
    return B[0, :] * emissivity * jnp.pi


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
    T = jnp.asarray(T, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    kappa = jnp.asarray(kappa, dtype=jnp.float64)
    W, mu = _gauss_quad(Gauss_quad)

    if surf_reflect is None:
        surf_reflect = jnp.zeros(wl.shape[0])
    surf_reflect = jnp.asarray(surf_reflect, dtype=jnp.float64)
    emissivity = 1.0 - surf_reflect
    B = planck_lambda_arr(T, wl)
    B = B.at[index_below_P_surf, :].set(B[index_below_P_surf, :] * emissivity)

    N_T = T.shape[0]
    N_wl = wl.shape[0]
    N_mu = mu.shape[0]

    dtau = kappa * dz[:, None]
    I0 = jnp.broadcast_to(B[0, :], (N_mu, N_wl))

    def scan_layer(I, i):
        Trans = jnp.exp(-dtau[i, :][None, :] / mu[:, None])
        I_new = Trans * I + (1.0 - Trans) * B[i, :][None, :]
        return I_new, None

    I_final, _ = lax.scan(scan_layer, I0, jnp.arange(N_T))
    F = jnp.sum(2.0 * jnp.pi * mu[:, None] * I_final * W[:, None], axis=0)
    return F, dtau


def determine_photosphere_radii(dtau, r_low, wl, photosphere_tau=2 / 3):
    """Wavelength-dependent photosphere radii via τ-interpolation.

    Bit-equivalent port of POSEIDON `emission.py:346-380`. Uses
    `jnp.interp` per wavelength.
    """
    dtau = jnp.asarray(dtau, dtype=jnp.float64)
    r_low = jnp.asarray(r_low, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    tau_cum = jnp.cumsum(dtau, axis=0)

    # Vectorise jnp.interp over wavelength.
    def one_wl(tau_lambda):
        return jnp.interp(photosphere_tau, tau_lambda, r_low)

    R_p_eff = jax.vmap(one_wl, in_axes=1)(tau_cum)
    return R_p_eff


def slice_gt(array, lim):
    """Clip per-row values above `lim` (POSEIDON `emission.py:423-432`)."""
    array = jnp.asarray(array, dtype=jnp.float64)
    return jnp.minimum(array, lim)


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

    A = jnp.zeros((L, N_wl))
    B = jnp.zeros((L, N_wl))
    C = jnp.zeros((L, N_wl))
    D = jnp.zeros((L, N_wl))

    # Row 0.
    A = A.at[0, :].set(0.0)
    B = B.at[0, :].set(gamma[0, :] + 1.0)
    C = C.at[0, :].set(gamma[0, :] - 1.0)
    D = D.at[0, :].set(b_top - c_minus_up[0, :])

    # Odd rows 1..L-3 (i.e. A[1::2][:-1]).
    odd_idx = jnp.arange(1, L - 1, 2)
    A = A.at[odd_idx, :].set((e1[:-1, :] + e3[:-1, :]) * (gamma[1:, :] - 1.0))
    B = B.at[odd_idx, :].set((e2[:-1, :] + e4[:-1, :]) * (gamma[1:, :] - 1.0))
    C = C.at[odd_idx, :].set(2.0 * (1.0 - gamma[1:, :] ** 2))
    D = D.at[odd_idx, :].set(
        (gamma[1:, :] - 1.0) * (c_plus_up[1:, :] - c_plus_down[:-1, :])
        + (1.0 - gamma[1:, :]) * (c_minus_down[:-1, :] - c_minus_up[1:, :])
    )

    # Even rows 2..L-2 (i.e. A[::2][1:]).
    even_idx = jnp.arange(2, L, 2)
    A = A.at[even_idx, :].set(2.0 * (1.0 - gamma[:-1, :] ** 2))
    B = B.at[even_idx, :].set((e1[:-1, :] - e3[:-1, :]) * (gamma[1:, :] + 1.0))
    C = C.at[even_idx, :].set((e1[:-1, :] + e3[:-1, :]) * (gamma[1:, :] - 1.0))
    D = D.at[even_idx, :].set(
        e3[:-1, :] * (c_plus_up[1:, :] - c_plus_down[:-1, :])
        + e1[:-1, :] * (c_minus_down[:-1, :] - c_minus_up[1:, :])
    )

    # Last row (L-1).
    A = A.at[L - 1, :].set(e1[-1, :] - surf_reflect * e3[-1, :])
    B = B.at[L - 1, :].set(e2[-1, :] - surf_reflect * e4[-1, :])
    C = C.at[L - 1, :].set(0.0)
    D = D.at[L - 1, :].set(
        b_surface - c_plus_down[-1, :] + surf_reflect * c_minus_down[-1, :]
    )

    return A, B, C, D


def _tri_diag_solve_scan(a, b, c, d):
    """Vectorised Thomas via two `lax.scan` sweeps.

    Single-wavelength formulation in POSEIDON `emission.py:534-569`. Each
    of `a, b, c, d` has shape `(L,)`; this routine returns `XK` of shape
    `(L,)`. For multi-wavelength batched solve, vmap over the trailing
    axis.
    """
    L = a.shape[0]
    AS_last = a[-1] / b[-1]
    DS_last = d[-1] / b[-1]

    # Backward sweep i = L-2 ... 0, carry (AS_prev, DS_prev).
    def backward_step(carry, idx):
        AS_next, DS_next = carry
        i = L - 2 - idx
        x = 1.0 / (b[i] - c[i] * AS_next)
        AS_i = a[i] * x
        DS_i = (d[i] - c[i] * DS_next) * x
        return (AS_i, DS_i), (AS_i, DS_i)

    (_, _), (AS_rev, DS_rev) = lax.scan(
        backward_step, (AS_last, DS_last), jnp.arange(L - 1)
    )
    # AS_rev[k] corresponds to i = L-2-k. Reverse to forward order:
    AS = jnp.concatenate([AS_rev[::-1], jnp.array([AS_last])])
    DS = jnp.concatenate([DS_rev[::-1], jnp.array([DS_last])])

    # Forward sweep i = 1..L-1, carry XK_{i-1}.
    XK_0 = DS[0]

    def forward_step(XK_prev, i):
        XK_i = DS[i] - AS[i] * XK_prev
        return XK_i, XK_i

    _, XK_rest = lax.scan(forward_step, XK_0, jnp.arange(1, L))
    XK = jnp.concatenate([jnp.array([XK_0]), XK_rest])
    return XK


def tri_diag_solve(l, a, b, c, d):
    """Tridiagonal Thomas-algorithm solve A·X = D for a single wavelength.

    Bit-equivalent port of POSEIDON `emission.py:534-569`. Uses `lax.scan`
    for the two sequential sweeps so the kernel is jit-traceable.
    """
    a = jnp.asarray(a, dtype=jnp.float64)
    b = jnp.asarray(b, dtype=jnp.float64)
    c = jnp.asarray(c, dtype=jnp.float64)
    d = jnp.asarray(d, dtype=jnp.float64)
    return _tri_diag_solve_scan(a, b, c, d)


def numba_cumsum(mat):
    """Column-wise cumulative sum.

    Bit-equivalent port of POSEIDON `emission.py:966-973`.
    """
    return jnp.cumsum(jnp.asarray(mat, dtype=jnp.float64), axis=0)


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
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    dtau_tot = jnp.asarray(dtau_tot, dtype=jnp.float64)
    kappa_Ray = jnp.asarray(kappa_Ray, dtype=jnp.float64)
    kappa_cloud = jnp.asarray(kappa_cloud, dtype=jnp.float64)
    kappa_tot = jnp.asarray(kappa_tot, dtype=jnp.float64)
    w_cloud = jnp.asarray(w_cloud, dtype=jnp.float64)
    g_cloud = jnp.asarray(g_cloud, dtype=jnp.float64)
    surf_reflect = jnp.asarray(surf_reflect, dtype=jnp.float64)
    kappa_cloud_seperate = jnp.asarray(kappa_cloud_seperate, dtype=jnp.float64)

    n_aerosol = kappa_cloud_seperate.shape[0]

    def loop_aerosol(carry, k):
        w_cloud_, kcws, kcwgs = carry
        w_cloud_ = w_cloud_.at[k, :, 0, zone_idx, :].set(
            w_cloud_[k, :, 0, zone_idx, :] * 0.99999
        )
        kcws = kcws.at[:, 0, zone_idx, :].add(
            kappa_cloud_seperate[k, :, 0, zone_idx, :] * w_cloud_[k, :, 0, zone_idx, :]
        )
        kcwgs = kcwgs.at[:, 0, zone_idx, :].add(
            kappa_cloud_seperate[k, :, 0, zone_idx, :]
            * w_cloud_[k, :, 0, zone_idx, :]
            * g_cloud[k, :, 0, zone_idx, :]
        )
        return (w_cloud_, kcws, kcwgs), None

    kappa_cloud_w_cloud_sum = jnp.zeros_like(kappa_cloud)
    kappa_cloud_w_cloud_g_cloud_sum = jnp.zeros_like(kappa_cloud)
    (w_cloud, kappa_cloud_w_cloud_sum, kappa_cloud_w_cloud_g_cloud_sum), _ = lax.scan(
        loop_aerosol,
        (w_cloud, kappa_cloud_w_cloud_sum, kappa_cloud_w_cloud_g_cloud_sum),
        jnp.arange(n_aerosol),
    )

    w_tot = (
        (0.99999 * kappa_Ray[:, 0, zone_idx, :])
        + (kappa_cloud_w_cloud_sum[:, 0, zone_idx, :])
    ) / kappa_tot
    g_tot = (kappa_cloud_w_cloud_g_cloud_sum[:, 0, zone_idx, :]) / (
        kappa_cloud_w_cloud_sum[:, 0, zone_idx, :]
        + (0.99999 * kappa_Ray[:, 0, zone_idx, :])
    )

    P = jnp.flipud(P)
    T = jnp.flipud(T)
    dtau_tot = jnp.flipud(dtau_tot)
    w_tot = jnp.flipud(w_tot)
    g_tot = jnp.flipud(g_tot)

    N_wl = wl.shape[0]
    N_layer = P.shape[0]
    N_level = N_layer + 1

    T_level = jnp.zeros(N_level)
    log_P_level = jnp.zeros(N_level)
    T_level = T_level.at[1:-1].set((T[1:] + T[:-1]) / 2.0)
    T_level = T_level.at[0].set(T_level[1] - (T_level[2] - T_level[1]))
    T_level = T_level.at[-1].set(T_level[-2] + (T_level[-2] - T_level[-3]))

    log_P = jnp.log10(P)
    log_P_level = log_P_level.at[1:-1].set((log_P[1:] + log_P[:-1]) / 2.0)
    log_P_level = log_P_level.at[0].set(
        log_P_level[1] - (log_P_level[2] - log_P_level[1])
    )
    log_P_level = log_P_level.at[-1].set(
        log_P_level[-2] + (log_P_level[-2] - log_P_level[-3])
    )
    P_level = jnp.power(10.0, log_P_level)

    gangle = jnp.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = jnp.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )

    cos_theta = 1.0
    mu1 = 0.5
    all_b = planck_lambda_arr(T_level, wl)
    b0 = all_b[0:-1, :]
    b1 = (all_b[1:, :] - b0) / dtau_tot

    g1 = 2.0 - (w_tot * (1 + g_tot))
    g2 = w_tot * (1 - g_tot)
    lamda = jnp.sqrt(g1**2 - g2**2)
    gamma = (g1 - lamda) / g2
    g1_plus_g2 = 1.0 / (g1 + g2)

    c_plus_up = 2 * jnp.pi * mu1 * (b0 + b1 * g1_plus_g2)
    c_minus_up = 2 * jnp.pi * mu1 * (b0 - b1 * g1_plus_g2)
    c_plus_down = 2 * jnp.pi * mu1 * (b0 + b1 * dtau_tot + b1 * g1_plus_g2)
    c_minus_down = 2 * jnp.pi * mu1 * (b0 + b1 * dtau_tot - b1 * g1_plus_g2)

    exptrm = lamda * dtau_tot
    exptrm = slice_gt(exptrm, 35.0)
    exptrm_positive = jnp.exp(exptrm)
    exptrm_minus = 1.0 / exptrm_positive

    tau_top = dtau_tot[0, :] * P_level[0] / (P_level[1] - P_level[0])
    b_top = (1.0 - jnp.exp(-tau_top / mu1)) * all_b[0, :] * jnp.pi
    if hard_surface:
        emissivity = 1.0 - surf_reflect
        b_surface = emissivity * all_b[-1, :] * jnp.pi
    else:
        b_surface = (all_b[-1, :] + b1[-1, :] * mu1) * jnp.pi

    A, Bm, Cm, Dm = setup_tri_diag(
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
    # Vectorised solve over wavelengths.
    X_all = jax.vmap(_tri_diag_solve_scan, in_axes=(1, 1, 1, 1))(
        A, Bm, Cm, Dm
    )  # (N_wl, L)
    X_all = X_all.T  # (L, N_wl)
    positive = X_all[::2, :] + X_all[1::2, :]
    negative = X_all[::2, :] - X_all[1::2, :]

    G = (1 / mu1 - lamda) * positive
    H = gamma * (lamda + 1 / mu1) * negative
    alpha1 = 2 * jnp.pi * (b0 + b1 * (g1_plus_g2 - mu1))
    alpha2 = 2 * jnp.pi * b1

    exptrm_positive_mdpt = jnp.exp(0.5 * exptrm)
    exptrm_minus_mdpt = 1 / exptrm_positive_mdpt

    int_at_top = jnp.zeros((Gauss_quad, numt, N_wl))

    def one_angle(ng, nt, iubar, int_at_top):
        if hard_surface:
            emissivity = 1.0 - surf_reflect
            int_plus_last = emissivity * all_b[-1, :] * 2 * jnp.pi
        else:
            int_plus_last = (all_b[-1, :] + b1[-1, :] * iubar) * 2 * jnp.pi

        exptrm_angle = jnp.exp(-dtau_tot / iubar)
        exptrm_angle_mdpt = jnp.exp(-0.5 * dtau_tot / iubar)

        # Sweep from bottom to top: ibot = N_layer-1 .. 0
        def body(int_plus_prev, itop):
            ibot = N_layer - 1 - itop
            int_plus_new = (
                int_plus_prev * exptrm_angle[ibot, :]
                + (G[ibot, :] / (lamda[ibot, :] * iubar - 1.0))
                * (exptrm_positive[ibot, :] * exptrm_angle[ibot, :] - 1.0)
                + (H[ibot, :] / (lamda[ibot, :] * iubar + 1.0))
                * (1.0 - exptrm_minus[ibot, :] * exptrm_angle[ibot, :])
                + alpha1[ibot, :] * (1.0 - exptrm_angle[ibot, :])
                + alpha2[ibot, :]
                * (iubar - (dtau_tot[ibot, :] + iubar) * exptrm_angle[ibot, :])
            )
            int_plus_mdpt_new = (
                int_plus_prev * exptrm_angle_mdpt[ibot, :]
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
            return int_plus_new, int_plus_mdpt_new

        # Use lax.scan accumulating int_plus_mdpt; ibot iteration from N_layer-1 down to 0
        _, int_plus_mdpt_stack = lax.scan(body, int_plus_last, jnp.arange(N_layer))
        # int_plus_mdpt_stack[itop=0] corresponds to ibot=N_layer-1 (deepest);
        # we want int_plus_mdpt[0, :] which is the top layer, i.e. ibot=0,
        # i.e. itop=N_layer-1. So:
        top_val = int_plus_mdpt_stack[N_layer - 1]
        return int_at_top.at[ng, nt, :].set(top_val)

    for ng in range(Gauss_quad):
        for nt in range(numt):
            iubar = gangle[ng]
            int_at_top = one_angle(ng, nt, iubar, int_at_top)

    F = jnp.sum(int_at_top[:, 0, :] * gweight[:, None], axis=0)
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

    Bit-equivalent port of POSEIDON `emission.py:976-1573`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    dtau_tot = jnp.asarray(dtau_tot, dtype=jnp.float64)
    kappa_Ray = jnp.asarray(kappa_Ray, dtype=jnp.float64)
    kappa_cloud = jnp.asarray(kappa_cloud, dtype=jnp.float64)
    kappa_tot = jnp.asarray(kappa_tot, dtype=jnp.float64)
    w_cloud = jnp.asarray(w_cloud, dtype=jnp.float64)
    g_cloud = jnp.asarray(g_cloud, dtype=jnp.float64)
    surf_reflect = jnp.asarray(surf_reflect, dtype=jnp.float64)
    kappa_cloud_seperate = jnp.asarray(kappa_cloud_seperate, dtype=jnp.float64)

    N_wl = wl.shape[0]
    N_layer = P.shape[0]

    n_aerosol = kappa_cloud_seperate.shape[0]

    kappa_cloud_w_cloud_sum = jnp.zeros_like(kappa_cloud)
    kappa_cloud_g_cloud_sum = jnp.zeros_like(kappa_cloud)
    kappa_cloud_w_cloud_g_cloud_sum = jnp.zeros_like(kappa_cloud)
    g_cloud_tot_weighted = jnp.zeros_like(kappa_cloud)

    def aerosol_step(carry, k):
        w_cloud_, kcws, kcgs, kcwgs, gctw = carry
        w_cloud_ = w_cloud_.at[k, :, 0, zone_idx, :].set(
            w_cloud_[k, :, 0, zone_idx, :] * 0.99999
        )
        kcws = kcws.at[:, 0, zone_idx, :].add(
            kappa_cloud_seperate[k, :, 0, zone_idx, :] * w_cloud_[k, :, 0, zone_idx, :]
        )
        kcgs = kcgs.at[:, 0, zone_idx, :].add(
            kappa_cloud_seperate[k, :, 0, zone_idx, :] * g_cloud[k, :, 0, zone_idx, :]
        )
        kcwgs = kcwgs.at[:, 0, zone_idx, :].add(
            kappa_cloud_seperate[k, :, 0, zone_idx, :]
            * w_cloud_[k, :, 0, zone_idx, :]
            * g_cloud[k, :, 0, zone_idx, :]
        )
        gctw = gctw.at[:, 0, zone_idx, :].add(
            (
                kappa_cloud_seperate[k, :, 0, zone_idx, :]
                / kappa_cloud[:, 0, zone_idx, :]
            )
            * g_cloud[k, :, 0, zone_idx, :]
        )
        return (w_cloud_, kcws, kcgs, kcwgs, gctw), None

    (
        (
            w_cloud,
            kappa_cloud_w_cloud_sum,
            kappa_cloud_g_cloud_sum,
            kappa_cloud_w_cloud_g_cloud_sum,
            g_cloud_tot_weighted,
        ),
        _,
    ) = lax.scan(
        aerosol_step,
        (
            w_cloud,
            kappa_cloud_w_cloud_sum,
            kappa_cloud_g_cloud_sum,
            kappa_cloud_w_cloud_g_cloud_sum,
            g_cloud_tot_weighted,
        ),
        jnp.arange(n_aerosol),
    )

    g_cloud_tot_weighted = jnp.nan_to_num(g_cloud_tot_weighted, nan=0.0)

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

    P = jnp.flipud(P)
    dtau_tot = jnp.flipud(dtau_tot)
    w_tot = jnp.flipud(w_tot)
    g_cloud_tot_weighted = jnp.flipud(g_cloud_tot_weighted)
    ftau_cld = jnp.flipud(ftau_cld)
    ftau_ray = jnp.flipud(ftau_ray)
    g_cloud_tot_weighted = g_cloud_tot_weighted[:, 0, zone_idx, :]

    tau = jnp.zeros((N_layer + 1, N_wl))
    tau = tau.at[1:, :].set(numba_cumsum(dtau_tot[:, :]))

    stream = 2
    f_deltaM = g_cloud_tot_weighted**stream
    w_dedd = w_tot * (1.0 - f_deltaM) / (1.0 - w_tot * f_deltaM)
    g_dedd = (g_cloud_tot_weighted - f_deltaM) / (1.0 - f_deltaM)
    dtau_dedd = dtau_tot * (1.0 - w_tot * f_deltaM)
    tau_dedd = jnp.zeros((N_layer + 1, N_wl))
    tau_dedd = tau_dedd.at[1:, :].set(numba_cumsum(dtau_dedd[:, :]))

    gangle = jnp.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = jnp.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )

    phase_angle = 0.0
    cos_theta = 1.0
    longitude = jnp.arcsin(
        (gangle - (cos_theta - 1.0) / (cos_theta + 1.0)) / (2.0 / (cos_theta + 1))
    )
    colatitude = jnp.arccos(0.0)
    f = jnp.sin(colatitude)
    ubar0 = jnp.outer(jnp.cos(longitude - phase_angle), f)
    ubar1 = jnp.outer(jnp.cos(longitude), f)
    F0PI = jnp.zeros(N_wl) + 1

    xint_at_top = jnp.zeros((Gauss_quad, numt, N_wl))

    sq3 = jnp.sqrt(3.0)
    if toon_coefficients == 1:
        g1 = (7 - w_dedd * (4 + 3 * ftau_cld * g_dedd)) / 4
        g2 = -(1 - w_dedd * (4 - 3 * ftau_cld * g_dedd)) / 4
    else:
        g1 = (sq3 * 0.5) * (2.0 - w_dedd * (1.0 + ftau_cld * g_dedd))
        g2 = (sq3 * w_dedd * 0.5) * (1.0 - ftau_cld * g_dedd)

    lamda = jnp.sqrt(g1**2 - g2**2)
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

            x_ = jnp.exp(-tau_dedd[:-1, :] / u0)
            c_minus_up = a_minus * x_
            c_plus_up = a_plus * x_
            x_ = jnp.exp(-tau_dedd[1:, :] / u0)
            c_minus_down = a_minus * x_
            c_plus_down = a_plus * x_

            exptrm = lamda * dtau_dedd
            exptrm = slice_gt(exptrm, 35.0)
            exptrm_positive = jnp.exp(exptrm)
            exptrm_minus = 1.0 / exptrm_positive

            b_surface = 0.0 + surf_reflect * u0 * F0PI * jnp.exp(-tau_dedd[-1, :] / u0)

            A, Bm, Cm, Dm = setup_tri_diag(
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

            X_all = jax.vmap(_tri_diag_solve_scan, in_axes=(1, 1, 1, 1))(
                A, Bm, Cm, Dm
            ).T
            positive = X_all[::2, :] + X_all[1::2, :]
            negative = X_all[::2, :] - X_all[1::2, :]

            xint_last = (
                positive[-1, :] * exptrm_positive[-1, :]
                + gama[-1, :] * negative[-1, :] * exptrm_minus[-1, :]
                + c_plus_down[-1, :]
            ) / jnp.pi

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
            else:  # multi_phase == 1
                multi_plus = 1.0 + 1.5 * ftau_cld * g_dedd * u1
                multi_minus = 1.0 - 1.5 * ftau_cld * g_dedd * u1

            G = positive * (multi_plus + gama * multi_minus) * w_dedd
            H = negative * (gama * multi_plus + multi_minus) * w_dedd
            A_ms = (multi_plus * c_plus_up + multi_minus * c_minus_up) * w_dedd
            G = G * 0.5 / jnp.pi
            H = H * 0.5 / jnp.pi
            A_ms = A_ms * 0.5 / jnp.pi

            g_forward = constant_forward * g_cloud_tot_weighted
            g_back = constant_back * g_cloud_tot_weighted

            p_single = ftau_cld * (
                (frac_a + frac_b * g_back**frac_c)
                * (1 - g_forward**2)
                / jnp.sqrt((1 + g_forward**2 + 2 * g_forward * cos_theta) ** 3)
                + (1 - (frac_a + frac_b * g_back**frac_c))
                * (1 - g_back**2)
                / jnp.sqrt((1 + g_back**2 + 2 * g_back * cos_theta) ** 3)
            ) + ftau_ray * (0.75 * (1 + cos_theta**2.0))

            # Reverse sweep i = N_layer-1 ... 0, carry xint_{i+1}.
            def xint_body(xint_prev, itop):
                i = N_layer - 1 - itop
                xint_new = (
                    xint_prev * jnp.exp(-dtau_dedd[i, :] / u1)
                    + (w_tot[i, :] * F0PI / (4.0 * jnp.pi))
                    * (p_single[i, :])
                    * jnp.exp(-tau[i, :] / u0)
                    * (1.0 - jnp.exp(-dtau_tot[i, :] * (u0 + u1) / (u0 * u1)))
                    * (u0 / (u0 + u1))
                    + A_ms[i, :]
                    * (1.0 - jnp.exp(-dtau_dedd[i, :] * (u0 + 1 * u1) / (u0 * u1)))
                    * (u0 / (u0 + 1 * u1))
                    + G[i, :]
                    * (jnp.exp(exptrm[i, :] * 1 - dtau_dedd[i, :] / u1) - 1.0)
                    / (lamda[i, :] * 1 * u1 - 1.0)
                    + H[i, :]
                    * (1.0 - jnp.exp(-exptrm[i, :] * 1 - dtau_dedd[i, :] / u1))
                    / (lamda[i, :] * 1 * u1 + 1.0)
                )
                return xint_new, xint_new

            _, xint_stack = lax.scan(xint_body, xint_last, jnp.arange(N_layer))
            xint_top = xint_stack[N_layer - 1]
            xint_at_top = xint_at_top.at[ng, nt, :].set(xint_top)

    tweight = jnp.array([1.0])
    sym_fac = 2 * jnp.pi
    albedo = jnp.zeros(N_wl)
    for ig in range(Gauss_quad):
        for it in range(int(tweight.shape[0])):
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
    """Construct ``(surf_reflect, surf_reflect_array)``.

    Bit-equivalent port of POSEIDON ``core.py:1741-1770``. Setup-time
    branching on string dispatch keys; setup-only, not called inside jit.
    """
    from jaxposeidon._surface_setup import interpolate_surface_components

    if surface or albedo_deck != -1:
        if surface:
            if surface_model == "gray":
                surf_reflect = jnp.zeros_like(wl)
                surf_reflect_array = []
            elif surface_model == "constant":
                surf_reflect = jnp.full_like(wl, albedo_surf)
                surf_reflect_array = []
            elif surface_model == "lab_data":
                surf_reflect_array = interpolate_surface_components(
                    wl, surface_components, surface_component_albedos
                )
                if surface_percentage_apply_to == "albedos":
                    surf_reflect = jnp.zeros_like(wl)
                    for n in range(len(surface_component_percentages)):
                        surf_reflect += (
                            surface_component_percentages[n] * surf_reflect_array[n]
                        )
                else:
                    surf_reflect = jnp.full_like(wl, -1.0)
        else:
            surf_reflect = jnp.full_like(wl, albedo_deck)
            surf_reflect_array = []
    else:
        surf_reflect = jnp.full_like(wl, -1.0)
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

    Port of POSEIDON `emission.py:1681-1878` (CPU subset). String-keyed
    setup dispatch is done in Python (setup-only); the leaf single-stream
    calls themselves are JAX-traceable.
    """
    from jaxposeidon._surface_setup import find_nearest_less_than

    if cloud_dim == 2:
        kappa_cloud_clear = jnp.zeros_like(kappa_cloud)
        kappa_tot_clear = (
            kappa_gas[:, 0, zone_idx, :]
            + kappa_Ray[:, 0, zone_idx, :]
            + kappa_cloud_clear[:, 0, zone_idx, :]
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
            F_p = jnp.zeros_like(wl)
            for n in range(len(surface_component_percentages)):
                F_p = F_p + surface_component_percentages[n] * F_p_array[n]

        if not disable_atmosphere:
            dtau = dtau_tot
        else:
            dtau = 0
    else:
        F_p, dtau = emission_single_stream(T, dz, wl, kappa_tot, Gauss_quad)
        dtau = jnp.flip(dtau, axis=0)
        if cloud_dim == 2:
            F_p_clear, dtau = emission_single_stream(
                T, dz, wl, kappa_tot_clear, Gauss_quad
            )
            F_p = (f_cloud * F_p) + ((1 - f_cloud) * F_p_clear)

    return F_p, dtau


def reflection_bare_surface(wl, surf_reflect, Gauss_quad=5):
    """Bare-rock reflected-light albedo with 5-pt Gaussian disk integration.

    Bit-equivalent port of POSEIDON `emission.py:1612-1700` simplified.
    """
    wl = jnp.asarray(wl, dtype=jnp.float64)
    surf_reflect = jnp.asarray(surf_reflect, dtype=jnp.float64)
    N_wl = wl.shape[0]
    gangle = jnp.array(
        [0.0985350858, 0.3045357266, 0.5620251898, 0.8019865821, 0.9601901429]
    )
    gweight = jnp.array(
        [0.0157479145, 0.0739088701, 0.1463869871, 0.1671746381, 0.0967815902]
    )
    tweight = jnp.array([1.0])
    cos_theta = 1.0
    F0PI = jnp.zeros(N_wl) + 1
    phase_angle = 0.0
    longitude = jnp.arcsin(
        (gangle - (cos_theta - 1.0) / (cos_theta + 1.0)) / (2.0 / (cos_theta + 1))
    )
    colatitude = jnp.arccos(0.0)
    f = jnp.sin(colatitude)
    ubar0 = jnp.outer(jnp.cos(longitude - phase_angle), f)

    sym_fac = 2 * jnp.pi
    xint_at_top = jnp.zeros((Gauss_quad, tweight.shape[0], N_wl))
    for ig in range(Gauss_quad):
        u0 = ubar0[ig, 0]
        xint_at_top = xint_at_top.at[ig, 0, :].set(surf_reflect * u0 * F0PI / jnp.pi)

    albedo = jnp.zeros(N_wl)
    for ig in range(Gauss_quad):
        for it in range(int(tweight.shape[0])):
            albedo = albedo + xint_at_top[ig, it, :] * gweight[ig] * tweight[it]
    albedo = sym_fac * 0.5 * albedo / F0PI * (cos_theta + 1.0)
    return albedo
