"""Atmosphere column construction.

Mirrors POSEIDON's `atmosphere.py:profiles(...)` (`atmosphere.py:2015-2493`).

Currently supported envelope:
- `PT_profile` in {`isotherm`, `Madhu`, `slope`, `Pelletier`, `Guillot`,
  `Guillot_dayside`, `Line`} with `PT_dim == 1`
- `X_profile == 'isochem'`, `X_dim == 1`, no per-species gradients
- `bulk_species`: `['H2']`, `['H2', 'He']`, `['H', 'He']`, or a single
  non-ghost species (H2/H/He dissociation mixture deferred to 0.5.7)
- `Atmosphere_dimension == 1`

Deferred to later phases:
- `gradient` / `two-gradients` PT profiles (0.5.9 — 2D/3D scaffolding)
- `file_read` PT/X profiles (0.5.17 — file I/O)
- non-isochem X_profile, dissociation, mu_back, Na_K (0.5.7)
- chem_eq grids (0.5.8)
- N_sectors / N_zones > 1 (0.5.9)

Numerical strategy: bit-equivalent port via float64 numpy and scipy's
`gaussian_filter1d`, `pchip_interpolate`, `special.expn`. JAX-native
versions are a v1 work item.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402  (used by setup-time helpers; hot-path uses jnp)
import scipy.constants as sc  # noqa: E402  (CODATA float constants only)

from jaxposeidon._jax_filters import gaussian_filter1d_edge  # noqa: E402
from jaxposeidon._jax_interpolate import pchip_interpolate  # noqa: E402
from jaxposeidon._jax_special import expn_2  # noqa: E402
from jaxposeidon._opacity_precompute import prior_index  # noqa: E402
from jaxposeidon._species_data import inactive_species as _INACT  # noqa: E402
from jaxposeidon._species_data import masses as _masses  # noqa: E402

_V0_PT_PROFILES = frozenset({"isotherm", "Madhu"})
_V05_PT_PROFILES = frozenset(
    {
        "isotherm",
        "Madhu",
        "slope",
        "Pelletier",
        "Guillot",
        "Guillot_dayside",
        "Line",
        "gradient",
        "two-gradients",
        "file_read",
    }
)
_V05_X_PROFILES = frozenset(
    {
        "isochem",
        "gradient",
        "two-gradients",
        "dissociation",
        "lever",
        "file_read",
        "chem_eq",
    }
)


# ---------------------------------------------------------------------------
# P-T profile builders (1D only)
# ---------------------------------------------------------------------------
def compute_T_isotherm(P, T_iso):
    """1D isothermal profile, shape (N_layers, 1, 1).

    Mirrors `POSEIDON/atmosphere.py:2152-2161` (no Gaussian smoothing for
    isotherm).
    """
    return T_iso * jnp.ones(shape=(len(P), 1, 1), dtype=jnp.float64)


def compute_T_Madhu(P, a1, a2, log_P1, log_P2, log_P3, T_set, P_set):
    """1D Madhusudhan & Seager (2009) P-T profile.

    Bit-equivalent to `POSEIDON/atmosphere.py:20-103`. Caller should
    Gaussian-smooth the output with `gauss_conv(...)` to match POSEIDON
    `atmosphere.py:2262`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    N_layers = P.shape[0]

    i_set = jnp.argmin(jnp.abs(P - P_set))
    P_set_i = P[i_set]

    log_P = jnp.log10(P)
    log_P_min = jnp.log10(jnp.min(P))
    log_P_set_i = jnp.log10(P_set_i)

    # Branch A: log_P_set_i >= log_P3
    T3_A = T_set
    T2_A = T3_A - ((1.0 / a2) * (log_P3 - log_P2)) ** 2
    T1_A = T2_A + ((1.0 / a2) * (log_P1 - log_P2)) ** 2
    T0_A = T1_A - ((1.0 / a1) * (log_P1 - log_P_min)) ** 2

    # Branch B: log_P_set_i >= log_P1 (and < log_P3)
    T2_B = T_set - ((1.0 / a2) * (log_P_set_i - log_P2)) ** 2
    T1_B = T2_B + ((1.0 / a2) * (log_P1 - log_P2)) ** 2
    T3_B = T2_B + ((1.0 / a2) * (log_P3 - log_P2)) ** 2
    T0_B = T1_B - ((1.0 / a1) * (log_P1 - log_P_min)) ** 2

    # Branch C: log_P_set_i < log_P1
    T0_C = T_set - ((1.0 / a1) * (log_P_set_i - log_P_min)) ** 2
    T1_C = T0_C + ((1.0 / a1) * (log_P1 - log_P_min)) ** 2
    T2_C = T1_C - ((1.0 / a2) * (log_P1 - log_P2)) ** 2
    T3_C = T2_C + ((1.0 / a2) * (log_P3 - log_P2)) ** 2

    is_A = log_P_set_i >= log_P3
    is_B = (~is_A) & (log_P_set_i >= log_P1)
    # else: branch C
    T0 = jnp.where(is_A, T0_A, jnp.where(is_B, T0_B, T0_C))
    T2 = jnp.where(is_A, T2_A, jnp.where(is_B, T2_B, T2_C))
    T3 = jnp.where(is_A, T3_A, jnp.where(is_B, T3_B, T3_C))

    # Per-layer profile (vectorised):
    T_top = T3
    T_mid = T2 + ((1.0 / a2) * (log_P - log_P2)) ** 2
    T_low = T0 + ((1.0 / a1) * (log_P - log_P_min)) ** 2
    is_top = log_P >= log_P3
    is_strict_mid = (log_P > log_P1) & (log_P < log_P3)
    T_per_layer = jnp.where(is_top, T_top, jnp.where(is_strict_mid, T_mid, T_low))
    return T_per_layer.reshape(N_layers, 1, 1)


def compute_T_Madhu_2D(
    P,
    a1_1,
    a2_1,
    log_P1_1,
    log_P2_1,
    a1_2,
    a2_2,
    log_P1_2,
    log_P2_2,
    T_deep,
    P_ref,
    N_sectors,
    N_zones,
    alpha,
    beta,
    phi,
    theta,
):
    """2D temperature field from two Madhu profiles sharing deep T at P_ref.

    Bit-equivalent port of POSEIDON `atmosphere.py:106-228`.
    """
    N_layers = len(P)
    log_P3 = np.log10(P_ref)

    T_profile_1 = compute_T_Madhu(
        P, a1_1, a2_1, log_P1_1, log_P2_1, log_P3, T_deep, P_ref
    )
    T_profile_2 = compute_T_Madhu(
        P, a1_2, a2_2, log_P1_2, log_P2_2, log_P3, T_deep, P_ref
    )
    T_1 = T_profile_1[:, 0, 0]
    T_2 = T_profile_2[:, 0, 0]

    T = np.zeros(shape=(N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)

    for j in range(N_sectors):
        for k in range(N_zones):
            if N_sectors > 1:
                if phi[j] <= -alpha_rad / 2.0:
                    w_phi = 1.0
                elif phi[j] >= alpha_rad / 2.0:
                    w_phi = 0.0
                else:
                    w_phi = 0.5 - (phi[j] / alpha_rad)
            else:
                w_phi = 0.5

            if N_zones > 1:
                if theta[k] <= -beta_rad / 2.0:
                    w_theta = 1.0
                elif theta[k] >= beta_rad / 2.0:
                    w_theta = 0.0
                else:
                    w_theta = 0.5 - (theta[k] / beta_rad)
            else:
                w_theta = 0.5

            w = max(0.0, min(1.0, w_phi + w_theta - 0.5))
            for i in range(N_layers):
                T[i, j, k] = w * T_1[i] + (1.0 - w) * T_2[i]
    return T


def gauss_conv(arr, sigma=3, axis=0, mode="nearest"):
    """Gaussian-smooth `arr` along `axis`, matching POSEIDON's scipy call.

    POSEIDON imports scipy.ndimage.gaussian_filter1d as gauss_conv
    (`POSEIDON/atmosphere.py:9`) and applies it as
    `gauss_conv(T_rough, sigma=3, axis=0, mode='nearest')` for Madhu/gradient
    profiles. Defaults match.

    v1-B: backed by `_jax_filters.gaussian_filter1d_edge`. Applies along
    `axis=0` only (POSEIDON only ever passes `axis=0`); other axes are
    not used by POSEIDON.
    """
    if mode != "nearest":
        raise NotImplementedError(
            f"gauss_conv mode={mode!r} not in POSEIDON usage (only 'nearest')"
        )
    arr_j = jnp.asarray(arr, dtype=jnp.float64)
    if axis != 0:
        arr_j = jnp.moveaxis(arr_j, axis, 0)
    orig_shape = arr_j.shape
    flat = arr_j.reshape(orig_shape[0], -1)
    cols = [
        gaussian_filter1d_edge(flat[:, c], float(sigma)) for c in range(flat.shape[1])
    ]
    out = jnp.stack(cols, axis=1).reshape(orig_shape)
    if axis != 0:
        out = jnp.moveaxis(out, 0, axis)
    return out


def compute_T_slope(
    P,
    T_phot,
    Delta_T_arr,
    log_P_phot=0.5,
    log_P_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
):
    """1D Piette & Madhusudhan (2021) `slope` P-T profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:232-299`. PCHIP
    interpolation over (log_P_points, T_points). The `log_P_phot` and
    `log_P_arr` are kwargs (Python-static); `prior_index(log_P_phot,
    log_P_arr)` resolves at trace time and the loop unrolls.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    Delta_T_arr = jnp.asarray(Delta_T_arr, dtype=jnp.float64)
    N_layers = P.shape[0]
    log_P_arr_np = np.asarray(log_P_arr)
    log_P_points = np.sort(np.append(log_P_arr_np, log_P_phot))
    N_T_points = len(log_P_points)

    i_phot = prior_index(log_P_phot, log_P_arr_np, 0)

    T_points_list = []
    for i in range(0, i_phot + 1):
        if i == 0:
            T_points_list.append(T_phot - jnp.sum(Delta_T_arr[i_phot::-1]))
        else:
            T_points_list.append(T_phot - jnp.sum(Delta_T_arr[i_phot : i - 1 : -1]))
    T_points_list.append(jnp.asarray(T_phot, dtype=jnp.float64))
    for i in range(i_phot + 2, N_T_points):
        T_points_list.append(T_phot + jnp.sum(Delta_T_arr[i_phot + 1 : i]))
    T_points = jnp.stack([jnp.asarray(x, dtype=jnp.float64) for x in T_points_list])

    T_flat = pchip_interpolate(
        jnp.asarray(log_P_points, dtype=jnp.float64), T_points, jnp.log10(P)
    )
    return T_flat.reshape(N_layers, 1, 1)


def compute_T_Pelletier(P, T_points):
    """1D Pelletier (2021) `spline` P-T profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:302-341`. PCHIP
    interpolation between knots evenly spaced in log-pressure between
    `min(log10 P)` and `max(log10 P)`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T_points = jnp.asarray(T_points, dtype=jnp.float64)
    N_layers = P.shape[0]
    log_P = jnp.log10(P)
    P_min = jnp.min(log_P)
    P_max = jnp.max(log_P)
    number_P_knots = T_points.shape[0]
    log_P_points = jnp.linspace(P_min, P_max, num=number_P_knots)
    T_flat = pchip_interpolate(log_P_points, T_points, log_P)
    return T_flat.reshape(N_layers, 1, 1)


def compute_T_Guillot(P, g, log_kappa_IR, log_gamma, T_int, T_equ):
    """1D Guillot (2010) terminator (f=0.25) P-T profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:344-402`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    kappa_IR = jnp.power(10.0, log_kappa_IR)
    gamma = jnp.power(10.0, log_gamma)
    tau = ((P * 1e6) * kappa_IR) / g
    T_irr = T_equ * jnp.sqrt(2.0)
    N_layers = P.shape[0]
    T_flat = (
        0.75 * T_int**4.0 * (2.0 / 3.0 + tau)
        + 0.75
        * T_irr**4.0
        / 4.0
        * (
            2.0 / 3.0
            + 1.0 / gamma / 3.0**0.5
            + (gamma / 3.0**0.5 - 1.0 / 3.0**0.5 / gamma)
            * jnp.exp(-gamma * tau * 3.0**0.5)
        )
    ) ** 0.25
    return T_flat.reshape(N_layers, 1, 1)


def compute_T_Guillot_dayside(P, g, log_kappa_IR, log_gamma, T_int, T_equ):
    """1D Guillot (2010) dayside (f=0.5) P-T profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:405-462`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    kappa_IR = jnp.power(10.0, log_kappa_IR)
    gamma = jnp.power(10.0, log_gamma)
    tau = ((P * 1e6) * kappa_IR) / g
    T_irr = T_equ * jnp.sqrt(2.0)
    N_layers = P.shape[0]
    T_flat = (
        0.75 * T_int**4.0 * (2.0 / 3.0 + tau)
        + 0.75
        * T_irr**4.0
        / 2.0
        * (
            2.0 / 3.0
            + 1.0 / gamma / 3.0**0.5
            + (gamma / 3.0**0.5 - 1.0 / 3.0**0.5 / gamma)
            * jnp.exp(-gamma * tau * 3.0**0.5)
        )
    ) ** 0.25
    return T_flat.reshape(N_layers, 1, 1)


def compute_T_Line(
    P, g, T_eq, log_kappa_IR, log_gamma, log_gamma_2, alpha, beta, T_int
):
    """1D Line (2013) double-channel P-T profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:465-532` (PLATON
    `set_from_radiative_solution` form, eqns 13-16 of Line+13).
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    kappa_IR = jnp.power(10.0, log_kappa_IR)
    gamma = jnp.power(10.0, log_gamma)
    gamma2 = jnp.power(10.0, log_gamma_2)
    T_irr = beta * T_eq
    tau = ((P * 1e6) * kappa_IR) / g
    N_layers = P.shape[0]

    def incoming(gam):
        return (
            3.0
            / 4
            * T_irr**4
            * (
                2.0 / 3
                + 2.0 / 3 / gam * (1 + (gam * tau / 2 - 1) * jnp.exp(-gam * tau))
                + 2.0 * gam / 3 * (1 - tau**2 / 2) * expn_2(gam * tau)
            )
        )

    e1 = incoming(gamma)
    e2 = incoming(gamma2)
    T_flat = (
        3.0 / 4 * T_int**4 * (2.0 / 3 + tau) + (1 - alpha) * e1 + alpha * e2
    ) ** 0.25
    return T_flat.reshape(N_layers, 1, 1)


# ---------------------------------------------------------------------------
# 3D P-T gradient profiles (MacDonald & Lewis 2022) — `compute_T_field_*`
# ---------------------------------------------------------------------------
def compute_T_field_gradient(
    P,
    T_bar_term,
    Delta_T_term,
    Delta_T_DN,
    T_deep,
    N_sectors,
    N_zones,
    alpha,
    beta,
    phi,
    theta,
    P_deep=10.0,
    P_high=1.0e-5,
):
    """3D P-T field gradient (MacDonald & Lewis 2022).

    Bit-equivalent port of POSEIDON `atmosphere.py:536-633`.
    """
    N_layers = len(P)
    T = np.zeros(shape=(N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    T_Evening = T_bar_term + Delta_T_term / 2.0
    T_Morning = T_bar_term - Delta_T_term / 2.0

    for j in range(N_sectors):
        if phi[j] <= -alpha_rad / 2.0:
            T_term = T_Evening
        elif (phi[j] > -alpha_rad / 2.0) and (phi[j] < alpha_rad / 2.0):
            T_term = T_bar_term - (phi[j] / (alpha_rad / 2.0)) * (Delta_T_term / 2.0)
        elif phi[j] >= -alpha_rad / 2.0:
            T_term = T_Morning

        T_Day = T_term + Delta_T_DN / 2.0
        T_Night = T_term - Delta_T_DN / 2.0

        for k in range(N_zones):
            if theta[k] <= -beta_rad / 2.0:
                T_high = T_Day
            elif (theta[k] > -beta_rad / 2.0) and (theta[k] < beta_rad / 2.0):
                T_high = T_term - (theta[k] / (beta_rad / 2.0)) * (Delta_T_DN / 2.0)
            elif theta[k] >= -beta_rad / 2.0:
                T_high = T_Night

            dT_dlogP = (T_deep - T_high) / (np.log10(P_deep / P_high))

            for i in range(N_layers):
                if P[i] <= P_high:
                    T[i, j, k] = T_high
                elif (P[i] > P_high) and (P[i] < P_deep):
                    T[i, j, k] = T_high + dT_dlogP * np.log10(P[i] / P_high)
                elif P[i] >= P_deep:
                    T[i, j, k] = T_deep
    return T


def compute_T_field_two_gradients(
    P,
    T_bar_term_high,
    T_bar_term_mid,
    Delta_T_term_high,
    Delta_T_term_mid,
    Delta_T_DN_high,
    Delta_T_DN_mid,
    log_P_mid,
    T_deep,
    N_sectors,
    N_zones,
    alpha,
    beta,
    phi,
    theta,
    P_deep=10.0,
    P_high=1.0e-5,
):
    """3D P-T field with two pressure-gradients connected at P_mid.

    Bit-equivalent port of POSEIDON `atmosphere.py:637-755`.
    """
    N_layers = len(P)
    T = np.zeros(shape=(N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)
    T_Evening_high = T_bar_term_high + Delta_T_term_high / 2.0
    T_Evening_mid = T_bar_term_mid + Delta_T_term_mid / 2.0
    T_Morning_high = T_bar_term_high - Delta_T_term_high / 2.0
    T_Morning_mid = T_bar_term_mid - Delta_T_term_mid / 2.0
    P_mid = np.power(10.0, log_P_mid)

    for j in range(N_sectors):
        if phi[j] <= -alpha_rad / 2.0:
            T_term_high = T_Evening_high
            T_term_mid = T_Evening_mid
        elif (phi[j] > -alpha_rad / 2.0) and (phi[j] < alpha_rad / 2.0):
            T_term_high = T_bar_term_high - (phi[j] / (alpha_rad / 2.0)) * (
                Delta_T_term_high / 2.0
            )
            T_term_mid = T_bar_term_mid - (phi[j] / (alpha_rad / 2.0)) * (
                Delta_T_term_mid / 2.0
            )
        elif phi[j] >= -alpha_rad / 2.0:
            T_term_high = T_Morning_high
            T_term_mid = T_Morning_mid

        T_Day_high = T_term_high + Delta_T_DN_high / 2.0
        T_Day_mid = T_term_mid + Delta_T_DN_mid / 2.0
        T_Night_high = T_term_high - Delta_T_DN_high / 2.0
        T_Night_mid = T_term_mid - Delta_T_DN_mid / 2.0

        for k in range(N_zones):
            if theta[k] <= -beta_rad / 2.0:
                T_high = T_Day_high
                T_mid = T_Day_mid
            elif (theta[k] > -beta_rad / 2.0) and (theta[k] < beta_rad / 2.0):
                T_high = T_term_high - (theta[k] / (beta_rad / 2.0)) * (
                    Delta_T_DN_high / 2.0
                )
                T_mid = T_term_mid - (theta[k] / (beta_rad / 2.0)) * (
                    Delta_T_DN_mid / 2.0
                )
            elif theta[k] >= -beta_rad / 2.0:
                T_high = T_Night_high
                T_mid = T_Night_mid

            dT_dlogP_1 = (T_mid - T_high) / (np.log10(P_mid / P_high))
            dT_dlogP_2 = (T_deep - T_mid) / (np.log10(P_deep / P_mid))

            for i in range(N_layers):
                if P[i] <= P_high:
                    T[i, j, k] = T_high
                elif (P[i] > P_high) and (P[i] < P_mid):
                    T[i, j, k] = T_high + dT_dlogP_1 * np.log10(P[i] / P_high)
                elif (P[i] >= P_mid) and (P[i] < P_deep):
                    T[i, j, k] = T_mid + dT_dlogP_2 * np.log10(P[i] / P_mid)
                elif P[i] >= P_deep:
                    T[i, j, k] = T_deep
    return T


# ---------------------------------------------------------------------------
# 3D X-field gradient profiles (MacDonald & Lewis 2022)
# ---------------------------------------------------------------------------
def compute_X_field_gradient(
    P,
    log_X_state,
    N_sectors,
    N_zones,
    param_species,
    species_has_profile,
    alpha,
    beta,
    phi,
    theta,
    P_deep=10.0,
    P_high=1.0e-5,
):
    """3D abundance field with single pressure gradient.

    Bit-equivalent port of POSEIDON `atmosphere.py:759-878`.
    """
    N_layers = len(P)
    N_param_species = len(param_species)
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)

    for q in range(N_param_species):
        log_X_bar_term, Delta_log_X_term, Delta_log_X_DN, log_X_deep = log_X_state[q, :]
        X_bar_term = np.power(10.0, log_X_bar_term)
        X_deep = np.power(10.0, log_X_deep)
        X_Evening = X_bar_term * np.power(10.0, Delta_log_X_term / 2.0)
        X_Morning = X_bar_term * np.power(10.0, -Delta_log_X_term / 2.0)

        for j in range(N_sectors):
            if phi[j] <= -alpha_rad / 2.0:
                X_term = X_Evening
            elif (phi[j] > -alpha_rad / 2.0) and (phi[j] < alpha_rad / 2.0):
                X_term = X_bar_term * np.power(
                    10.0, -(phi[j] / (alpha_rad / 2.0)) * (Delta_log_X_term / 2.0)
                )
            elif phi[j] >= -alpha_rad / 2.0:
                X_term = X_Morning

            X_Day = X_term * np.power(10.0, Delta_log_X_DN / 2.0)
            X_Night = X_term * np.power(10.0, -Delta_log_X_DN / 2.0)

            for k in range(N_zones):
                if theta[k] <= -beta_rad / 2.0:
                    X_high = X_Day
                elif (theta[k] > -beta_rad / 2.0) and (theta[k] < beta_rad / 2.0):
                    X_high = X_term * np.power(
                        10.0, -(theta[k] / (beta_rad / 2.0)) * (Delta_log_X_DN / 2.0)
                    )
                elif theta[k] >= -beta_rad / 2.0:
                    X_high = X_Night

                if species_has_profile[q] == 1:
                    dlog_X_dlogP = np.log10(X_deep / X_high) / np.log10(P_deep / P_high)
                    for i in range(N_layers):
                        if P[i] <= P_high:
                            X_profiles[q, i, j, k] = X_high
                        elif (P[i] > P_high) and (P[i] < P_deep):
                            X_profiles[q, i, j, k] = X_high * np.power(
                                P[i] / P_high, dlog_X_dlogP
                            )
                        elif P[i] >= P_deep:
                            X_profiles[q, i, j, k] = X_deep
                else:
                    X_profiles[q, :, j, k] = X_high
    return X_profiles


def compute_X_field_two_gradients(
    P,
    log_X_state,
    N_sectors,
    N_zones,
    param_species,
    species_has_profile,
    alpha,
    beta,
    phi,
    theta,
    P_deep=10.0,
    P_high=1.0e-5,
):
    """3D abundance field with two pressure gradients.

    Bit-equivalent port of POSEIDON `atmosphere.py:882-1013`.
    """
    N_layers = len(P)
    N_param_species = len(param_species)
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)

    for q in range(N_param_species):
        (
            log_X_bar_term_high,
            log_X_bar_term_mid,
            Delta_log_X_term_high,
            Delta_log_X_term_mid,
            Delta_log_X_DN_high,
            Delta_log_X_DN_mid,
            log_P_X_mid,
            log_X_deep,
        ) = log_X_state[q, :]
        X_bar_term_high = np.power(10.0, log_X_bar_term_high)
        X_bar_term_mid = np.power(10.0, log_X_bar_term_mid)
        P_mid = np.power(10.0, log_P_X_mid)
        X_deep = np.power(10.0, log_X_deep)

        X_Evening_high = X_bar_term_high * np.power(10.0, Delta_log_X_term_high / 2.0)
        X_Evening_mid = X_bar_term_mid * np.power(10.0, Delta_log_X_term_mid / 2.0)
        X_Morning_high = X_bar_term_high * np.power(10.0, -Delta_log_X_term_high / 2.0)
        X_Morning_mid = X_bar_term_mid * np.power(10.0, -Delta_log_X_term_mid / 2.0)

        for j in range(N_sectors):
            if phi[j] <= -alpha_rad / 2.0:
                X_term_high = X_Evening_high
                X_term_mid = X_Evening_mid
            elif (phi[j] > -alpha_rad / 2.0) and (phi[j] < alpha_rad / 2.0):
                X_term_high = X_bar_term_high * np.power(
                    10.0, -(phi[j] / (alpha_rad / 2.0)) * (Delta_log_X_term_high / 2.0)
                )
                X_term_mid = X_bar_term_mid * np.power(
                    10.0, -(phi[j] / (alpha_rad / 2.0)) * (Delta_log_X_term_mid / 2.0)
                )
            elif phi[j] >= -alpha_rad / 2.0:
                X_term_high = X_Morning_high
                X_term_mid = X_Morning_mid

            X_Day_high = X_term_high * np.power(10.0, Delta_log_X_DN_high / 2.0)
            X_Day_mid = X_term_mid * np.power(10.0, Delta_log_X_DN_mid / 2.0)
            X_Night_high = X_term_high * np.power(10.0, -Delta_log_X_DN_high / 2.0)
            X_Night_mid = X_term_mid * np.power(10.0, -Delta_log_X_DN_mid / 2.0)

            for k in range(N_zones):
                if theta[k] <= -beta_rad / 2.0:
                    X_high = X_Day_high
                    X_mid = X_Day_mid
                elif (theta[k] > -beta_rad / 2.0) and (theta[k] < beta_rad / 2.0):
                    X_high = X_term_high * np.power(
                        10.0,
                        -(theta[k] / (beta_rad / 2.0)) * (Delta_log_X_DN_high / 2.0),
                    )
                    X_mid = X_term_mid * np.power(
                        10.0,
                        -(theta[k] / (beta_rad / 2.0)) * (Delta_log_X_DN_mid / 2.0),
                    )
                elif theta[k] >= -beta_rad / 2.0:
                    X_high = X_Night_high
                    X_mid = X_Night_mid

                if species_has_profile[q] == 1:
                    dlog_X_dlogP_1 = np.log10(X_mid / X_high) / np.log10(P_mid / P_high)
                    dlog_X_dlogP_2 = np.log10(X_deep / X_mid) / np.log10(P_deep / P_mid)
                    for i in range(N_layers):
                        if P[i] <= P_high:
                            X_profiles[q, i, j, k] = X_high
                        elif (P[i] > P_high) and (P[i] < P_mid):
                            X_profiles[q, i, j, k] = X_high * np.power(
                                P[i] / P_high, dlog_X_dlogP_1
                            )
                        elif (P[i] >= P_mid) and (P[i] < P_deep):
                            X_profiles[q, i, j, k] = X_mid * np.power(
                                P[i] / P_mid, dlog_X_dlogP_2
                            )
                        elif P[i] >= P_deep:
                            X_profiles[q, i, j, k] = X_deep
                else:
                    X_profiles[q, :, j, k] = X_high
    return X_profiles


# ---------------------------------------------------------------------------
# Parmentier 2018 thermal dissociation
# ---------------------------------------------------------------------------
_PARMENTIER_COEFFICIENTS = {
    "H2": (1.0, 2.41e4, 6.5, 10**-0.1),
    "H2O": (2.0, 4.83e4, 15.9, 10**-3.3),
    "TiO": (1.6, 5.94e4, 23.0, 10**-7.1),
    "VO": (1.5, 5.40e4, 23.8, 10**-9.2),
    "H-": (0.6, -0.14e4, 7.7, 10**-8.3),
    "Na": (0.6, 1.89e4, 12.2, 10**-5.5),
    "K": (0.6, 1.28e4, 12.7, 10**-7.1),
}


def Parmentier_dissociation_profile(P, T, A_0, alpha, beta, gamma, A_0_ref):
    """Parmentier+18 thermal-dissociation abundance profile.

    Bit-equivalent port of POSEIDON `atmosphere.py:1017-1052`.
    """
    log_A_shift = np.log10(A_0 / A_0_ref)
    A_d = (
        10 ** (log_A_shift - gamma)
        * P.astype("float64") ** alpha
        * 10 ** (beta / T.astype("float64"))
    )
    return ((1 / A_0) ** 0.5 + (1 / A_d) ** 0.5) ** (-2)


def compute_X_dissociation(
    P,
    T,
    log_X_state,
    N_sectors,
    N_zones,
    param_species,
    species_has_profile,
    alpha,
    beta,
    phi,
    theta,
):
    """3D abundance field with Parmentier+18 thermal dissociation.

    Bit-equivalent port of POSEIDON `atmosphere.py:1056-1166`. Per-species
    Parmentier coefficients in `_PARMENTIER_COEFFICIENTS`.
    """
    N_layers = len(P)
    N_param_species = len(param_species)
    X_profiles = np.zeros(shape=(N_param_species, N_layers, N_sectors, N_zones))
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)

    dissoc_species = ("H2O", "TiO", "VO", "H-", "Na", "K")

    for q in range(N_param_species):
        log_X_bar_term, Delta_log_X_term, Delta_log_X_DN = log_X_state[q, :]
        X_bar_term = np.power(10.0, log_X_bar_term)
        X_Evening = X_bar_term * np.power(10.0, Delta_log_X_term / 2.0)
        X_Morning = X_bar_term * np.power(10.0, -Delta_log_X_term / 2.0)

        for j in range(N_sectors):
            if phi[j] <= -alpha_rad / 2.0:
                X_term = X_Evening
            elif (phi[j] > -alpha_rad / 2.0) and (phi[j] < alpha_rad / 2.0):
                X_term = X_bar_term * np.power(
                    10.0, -(phi[j] / (alpha_rad / 2.0)) * (Delta_log_X_term / 2.0)
                )
            elif phi[j] >= -alpha_rad / 2.0:
                X_term = X_Morning

            X_Day = X_term * np.power(10.0, Delta_log_X_DN / 2.0)
            X_Night = X_term * np.power(10.0, -Delta_log_X_DN / 2.0)

            for k in range(N_zones):
                if theta[k] <= -beta_rad / 2.0:
                    X_deep = X_Day
                elif (theta[k] > -beta_rad / 2.0) and (theta[k] < beta_rad / 2.0):
                    X_deep = X_term * np.power(
                        10.0, -(theta[k] / (beta_rad / 2.0)) * (Delta_log_X_DN / 2.0)
                    )
                elif theta[k] >= -beta_rad / 2.0:
                    X_deep = X_Night

                if species_has_profile[q] == 1:
                    if param_species[q] in dissoc_species:
                        a_P, b_P, g_P, A0_ref_P = _PARMENTIER_COEFFICIENTS[
                            param_species[q]
                        ]
                        X_profiles[q, :, j, k] = Parmentier_dissociation_profile(
                            P, T[:, j, k], X_deep, a_P, b_P, g_P, A0_ref_P
                        )
                    else:
                        X_profiles[q, :, j, k] = X_deep
                else:
                    X_profiles[q, :, j, k] = X_deep
    return X_profiles


def compute_X_lever(P, log_X_state, species_has_profile, N_sectors, N_zones):
    """Lever-arm vertical mixing-ratio profile (1D only).

    Bit-equivalent port of POSEIDON `atmosphere.py:1169-1219`.
    """
    log_p = np.log10(P)
    N_param_species = np.shape(log_X_state)[0]
    log_X = np.zeros(shape=(N_param_species, len(P), N_sectors, N_zones))

    for q in range(N_param_species):
        log_X_q, log_P_q, upsilon_q = log_X_state[q, :]
        upsilon_q = upsilon_q * (np.pi / 180.0)

        if np.abs(upsilon_q) <= np.pi / 2:
            for j in range(len(log_p)):
                if log_p[j] <= log_P_q:
                    log_X[q, j, 0, 0] = log_X_q + np.tan(upsilon_q) * (
                        log_p[j] - log_P_q
                    )
                elif log_p[j] > log_P_q:
                    log_X[q, j, 0, 0] = log_X_q
        elif (np.abs(upsilon_q) > np.pi / 2) and (np.abs(upsilon_q) <= np.pi):
            for j in range(len(log_p)):
                if log_p[j] > log_P_q:
                    log_X[q, j, 0, 0] = log_X_q + np.tan(upsilon_q) * (
                        log_p[j] - log_P_q
                    )
                elif log_p[j] <= log_P_q:
                    log_X[q, j, 0, 0] = log_X_q
    return np.power(10, log_X)


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
        X = 10.0**log_X_bar_term
        X_profiles[q, :, :, :] = X
    return X_profiles


def add_bulk_component(
    P, T, X_param, N_species, N_sectors, N_zones, bulk_species, He_fraction
):
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
        X_bulk = 1.0 - np.sum(X_param, axis=0)
        X_H2_deep = X_bulk / (1.0 + He_fraction)
        a_P, b_P, g_P, A0_ref_P = _PARMENTIER_COEFFICIENTS["H2"]
        for j in range(N_sectors):
            for k in range(N_zones):
                X_H2 = Parmentier_dissociation_profile(
                    P, T[:, j, k], X_H2_deep[:, j, k], a_P, b_P, g_P, A0_ref_P
                )
                X_H = (X_bulk[:, j, k] - (1 + He_fraction) * X_H2) / (
                    1 + He_fraction / 2.0
                )
                X_He = (He_fraction * X_H2) + ((He_fraction / 2.0) * X_H)
                X[0, :, j, k] = X_H2
                X[1, :, j, k] = X_H
                X[2, :, j, k] = X_He
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
    """Mean molecular mass profile in kg.

    Mirrors POSEIDON `atmosphere.py:1816-1856`. Same numba-broadcast
    workaround as `radial_profiles` when X is shaped (N, len(P), 1, 1)
    while caller requests N_sectors > 1 or N_zones > 1.
    """
    N_layers = len(P)
    if X.shape != (N_species, N_layers, N_sectors, N_zones):
        X = np.broadcast_to(X, (N_species, N_layers, N_sectors, N_zones))
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
def _radial_column(P, T_col, mu_col, g_0, R_p, P_ref, R_p_ref, constant_g=False):
    """Single-(sector, zone) hydrostatic radius integral.

    Computes (n, r, r_up, r_low, dr) for one (j, k) atmospheric column.
    Used by both `radial_profiles` (inverse-square gravity) and
    `radial_profiles_constant_g`.
    """
    N_layers = P.shape[0]
    log_P = jnp.log(P)
    n = (P * 1.0e5) / (sc.k * T_col)
    if constant_g:
        integrand = (sc.k * T_col) / (g_0 * mu_col)
    else:
        integrand = (sc.k * T_col) / (R_p**2 * g_0 * mu_col)

    i_ref = jnp.argmin(jnp.abs(P - P_ref))

    # Forward cumulative trapezoid from i_ref+1 onward.
    delta = 0.5 * (integrand[1:] + integrand[:-1]) * (log_P[1:] - log_P[:-1])
    # Add zero at position 0 to align as "increment from layer i-1 to i".
    delta_padded = jnp.concatenate([jnp.zeros(1, dtype=delta.dtype), delta])
    # Outward integral: cumulative sum forward, taking layers > i_ref.
    cum_fwd = jnp.cumsum(delta_padded)
    # Inward integral: cumulative sum backward.
    # integral_in[i] = sum_{m=i_ref-1 down to i} 0.5*(integrand[m]+integrand[m+1])*(log_P[m]-log_P[m+1])
    delta_in = 0.5 * (integrand[:-1] + integrand[1:]) * (log_P[:-1] - log_P[1:])
    delta_in_padded = jnp.concatenate([delta_in, jnp.zeros(1, dtype=delta.dtype)])
    cum_in = jnp.cumsum(delta_in_padded[::-1])[::-1]

    # cum_fwd[i_ref] = sum_{m=1..i_ref}, but we want integral_out at i =
    # sum_{m=i_ref+1..i}. Compute via cum_fwd - cum_fwd[i_ref].
    integral_out = cum_fwd - cum_fwd[i_ref]
    # cum_in[i] = sum from i to N-2. integral_in at i = sum from
    # m=i to i_ref-1 = cum_in[i] - cum_in[i_ref].
    integral_in = cum_in - cum_in[i_ref]

    idx = jnp.arange(N_layers)
    if constant_g:
        r_out = R_p_ref - integral_out
        r_in = R_p_ref - integral_in
    else:
        r_out = 1.0 / ((1.0 / R_p_ref) + integral_out)
        r_in = 1.0 / ((1.0 / R_p_ref) + integral_in)
    r = jnp.where(idx > i_ref, r_out, jnp.where(idx < i_ref, r_in, R_p_ref))

    # Layer-boundary radii and thicknesses.
    r_up_mid = 0.5 * (r[2:] + r[1:-1])
    r_low_mid = 0.5 * (r[1:-1] + r[:-2])
    dr_mid = 0.5 * (r[2:] - r[:-2])

    r_up_0 = 0.5 * (r[1] + r[0])
    r_up_last = r[-1] + 0.5 * (r[-1] - r[-2])
    r_low_0 = r[0] - 0.5 * (r[1] - r[0])
    r_low_last = 0.5 * (r[-1] + r[-2])
    dr_0 = r[1] - r[0]
    dr_last = r[-1] - r[-2]

    r_up = jnp.concatenate(
        [jnp.atleast_1d(r_up_0), r_up_mid, jnp.atleast_1d(r_up_last)]
    )
    r_low = jnp.concatenate(
        [jnp.atleast_1d(r_low_0), r_low_mid, jnp.atleast_1d(r_low_last)]
    )
    dr = jnp.concatenate([jnp.atleast_1d(dr_0), dr_mid, jnp.atleast_1d(dr_last)])
    return n, r, r_up, r_low, dr


def radial_profiles(P, T, g_0, R_p, P_ref, R_p_ref, mu, N_sectors, N_zones):
    """Inverse-square-gravity hydrostatic radius profile.

    Mirrors `POSEIDON/atmosphere.py:1494-1606`. Outer (j, k) loops
    unroll at trace time (static `N_sectors`, `N_zones`); per-column
    integral is vectorised via `_radial_column` (jit-traceable).
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    N_layers = P.shape[0]
    if T.shape != (N_layers, N_sectors, N_zones):
        T = jnp.broadcast_to(T, (N_layers, N_sectors, N_zones))
    if mu.shape != (N_layers, N_sectors, N_zones):
        mu = jnp.broadcast_to(mu, (N_layers, N_sectors, N_zones))

    n_list = []
    r_list = []
    r_up_list = []
    r_low_list = []
    dr_list = []
    for j in range(N_sectors):
        for k in range(N_zones):
            ncol, rcol, rup, rlow, drcol = _radial_column(
                P,
                T[:, j, k],
                mu[:, j, k],
                g_0,
                R_p,
                P_ref,
                R_p_ref,
                constant_g=False,
            )
            n_list.append(ncol)
            r_list.append(rcol)
            r_up_list.append(rup)
            r_low_list.append(rlow)
            dr_list.append(drcol)

    def _stack(lst):
        out = jnp.stack(lst, axis=-1)
        return out.reshape(N_layers, N_sectors, N_zones)

    return (
        _stack(n_list),
        _stack(r_list),
        _stack(r_up_list),
        _stack(r_low_list),
        _stack(dr_list),
    )


# ---------------------------------------------------------------------------
# Mixing-ratio categorisation
# ---------------------------------------------------------------------------
# POSEIDON `supported_chemicals.inactive_species` — sourced from the local
# build-time-extracted `_species_data.inactive_species` so the runtime
# forward path is independent of POSEIDON's importability.
_INACTIVE_SPECIES = _INACT


def mixing_ratio_categories(
    P,
    X,
    N_sectors,
    N_zones,
    included_species,
    active_species,
    CIA_pairs,
    ff_pairs,
    bf_species,
):
    """Slice the full X array into active / CIA / ff / bf categories.

    Mirrors POSEIDON `atmosphere.py:1722-1813`. Used by `profiles(...)`
    to package the atmosphere tuple for the extinction stage.
    """
    N_layers = len(P)
    N_species_active = len(active_species)
    N_CIA_pairs = len(CIA_pairs)
    N_ff_pairs = len(ff_pairs)
    N_bf_species = len(bf_species)

    included = np.asarray(included_species)
    inactive_mask = np.isin(included, _INACTIVE_SPECIES)

    if X.shape[2:] != (N_sectors, N_zones):
        X = np.broadcast_to(X, (X.shape[0], N_layers, N_sectors, N_zones))

    X_active = X[~inactive_mask, :, :, :]
    X_CIA = np.zeros((2, N_CIA_pairs, N_layers, N_sectors, N_zones))
    X_ff = np.zeros((2, N_ff_pairs, N_layers, N_sectors, N_zones))
    X_bf = np.zeros((N_bf_species, N_layers, N_sectors, N_zones))

    for q in range(N_CIA_pairs):
        pair = CIA_pairs[q]
        a, b = pair.split("-")
        X_CIA[0, q, :, :, :] = X[included == a, :, :, :]
        X_CIA[1, q, :, :, :] = X[included == b, :, :, :]

    for q in range(N_ff_pairs):
        pair = ff_pairs[q]
        if pair == "H-ff":
            X_ff[0, q, :, :, :] = X[included == "H", :, :, :]
            X_ff[1, q, :, :, :] = X[included == "e-", :, :, :]

    for q in range(N_bf_species):
        species = bf_species[q]
        if species == "H-bf":
            X_bf[q, :, :, :] = X[included == "H-", :, :, :]

    return X_active, X_CIA, X_ff, X_bf


def radial_profiles_constant_g(P, T, g_0, P_ref, R_p_ref, mu, N_sectors, N_zones):
    """Constant-gravity hydrostatic radius profile.

    Mirrors `POSEIDON/atmosphere.py:1609-1722`. Implementation shares
    `_radial_column` with `radial_profiles` via `constant_g=True`.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    T = jnp.asarray(T, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    N_layers = P.shape[0]
    if T.shape != (N_layers, N_sectors, N_zones):
        T = jnp.broadcast_to(T, (N_layers, N_sectors, N_zones))
    if mu.shape != (N_layers, N_sectors, N_zones):
        mu = jnp.broadcast_to(mu, (N_layers, N_sectors, N_zones))

    n_list = []
    r_list = []
    r_up_list = []
    r_low_list = []
    dr_list = []
    for j in range(N_sectors):
        for k in range(N_zones):
            ncol, rcol, rup, rlow, drcol = _radial_column(
                P,
                T[:, j, k],
                mu[:, j, k],
                g_0,
                0.0,
                P_ref,
                R_p_ref,
                constant_g=True,
            )
            n_list.append(ncol)
            r_list.append(rcol)
            r_up_list.append(rup)
            r_low_list.append(rlow)
            dr_list.append(drcol)

    def _stack(lst):
        out = jnp.stack(lst, axis=-1)
        return out.reshape(N_layers, N_sectors, N_zones)

    return (
        _stack(n_list),
        _stack(r_list),
        _stack(r_up_list),
        _stack(r_low_list),
        _stack(dr_list),
    )


# ---------------------------------------------------------------------------
# profiles() — v0 top-level dispatcher
# ---------------------------------------------------------------------------
def profiles(
    P,
    R_p,
    g_0,
    PT_profile,
    X_profile,
    PT_state,
    P_ref,
    R_p_ref,
    log_X_state,
    included_species,
    bulk_species,
    param_species,
    active_species,
    CIA_pairs,
    ff_pairs,
    bf_species,
    N_sectors,
    N_zones,
    alpha,
    beta,
    phi,
    theta,
    species_vert_gradient,
    He_fraction,
    T_input=None,
    X_input=None,
    P_param_set=1.0e-6,
    log_P_slope_phot=0.5,
    log_P_slope_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
    Na_K_fixed_ratio=False,
    constant_gravity=False,
    chemistry_grid=None,
    PT_penalty=False,
    T_eq=None,
    mu_back=None,
    disable_atmosphere=False,
):
    """Port of POSEIDON `atmosphere.py:profiles(...)` (2015-2493).

    Returns the same 13-tuple POSEIDON does:
        (T, n, r, r_up, r_low, dr, mu, X, X_active, X_CIA, X_ff, X_bf, physical)
    where `physical` is True/False indicating whether the configuration
    produced a valid atmosphere.

    Supported envelope:
      - disable_atmosphere=False
      - PT_profile in {`isotherm`, `Madhu`, `slope`, `Pelletier`, `Guillot`,
        `Guillot_dayside`, `Line`} (1D only)
      - X_profile == 'isochem', X_dim=1, no species gradients
      - bulk_species per `add_bulk_component`
      - N_sectors == N_zones == 1
    Any other configuration raises NotImplementedError pointing at the
    phase where it lands.
    """
    if disable_atmosphere:
        raise NotImplementedError("disable_atmosphere=True deferred to Phase 0.5.13d")
    if PT_profile not in _V05_PT_PROFILES:
        raise NotImplementedError(
            f"PT_profile={PT_profile!r} not in v0.5 set ({sorted(_V05_PT_PROFILES)})"
        )
    if X_profile not in _V05_X_PROFILES:
        raise NotImplementedError(
            f"X_profile={X_profile!r} not in v0.5 set ({sorted(_V05_X_PROFILES)})"
        )
    if X_profile == "chem_eq" and chemistry_grid is None:
        raise Exception("Error: no chemistry grid loaded for an equilibrium model")

    if PT_profile == "isotherm":
        T = compute_T_isotherm(P, float(PT_state[0]))
    elif PT_profile == "gradient":
        T_bar_term, Delta_T_term, Delta_T_DN, T_deep = PT_state
        if (Delta_T_term < 0.0) or (Delta_T_DN < 0.0):
            return (0,) * 12 + (False,)
        T_rough = compute_T_field_gradient(
            P,
            T_bar_term,
            Delta_T_term,
            Delta_T_DN,
            T_deep,
            N_sectors,
            N_zones,
            alpha,
            beta,
            phi,
            theta,
        )
        T = gauss_conv(T_rough, sigma=3, axis=0, mode="nearest")
    elif PT_profile == "two-gradients":
        (
            T_bar_term_high,
            T_bar_term_mid,
            Delta_T_term_high,
            Delta_T_term_mid,
            Delta_T_DN_high,
            Delta_T_DN_mid,
            log_P_mid,
            T_deep,
        ) = PT_state
        if (
            (Delta_T_term_high < 0.0)
            or (Delta_T_DN_high < 0.0)
            or (Delta_T_term_mid < 0.0)
            or (Delta_T_DN_mid < 0.0)
        ):
            return (0,) * 12 + (False,)
        T_rough = compute_T_field_two_gradients(
            P,
            T_bar_term_high,
            T_bar_term_mid,
            Delta_T_term_high,
            Delta_T_term_mid,
            Delta_T_DN_high,
            Delta_T_DN_mid,
            log_P_mid,
            T_deep,
            N_sectors,
            N_zones,
            alpha,
            beta,
            phi,
            theta,
        )
        T = gauss_conv(T_rough, sigma=3, axis=0, mode="nearest")
    elif PT_profile == "Madhu":
        if len(PT_state) == 6:
            a1, a2, log_P1, log_P2, log_P3, T_set = PT_state
            if (log_P3 < log_P2) or (log_P3 < log_P1):
                return (0,) * 12 + (False,)
            T_rough = compute_T_Madhu(
                P, a1, a2, log_P1, log_P2, log_P3, T_set, P_param_set
            )
        elif len(PT_state) == 9:
            a1_1, a2_1, log_P1_1, log_P2_1, a1_2, a2_2, log_P1_2, log_P2_2, T_deep = (
                PT_state
            )
            log_P3 = np.log10(P_ref)
            if (
                (log_P3 < log_P2_1)
                or (log_P3 < log_P1_1)
                or (log_P3 < log_P2_2)
                or (log_P3 < log_P1_2)
            ):
                return (0,) * 12 + (False,)
            T_rough = compute_T_Madhu_2D(
                P,
                a1_1,
                a2_1,
                log_P1_1,
                log_P2_1,
                a1_2,
                a2_2,
                log_P1_2,
                log_P2_2,
                T_deep,
                P_ref,
                N_sectors,
                N_zones,
                alpha,
                beta,
                phi,
                theta,
            )
        else:
            raise ValueError(
                f"Madhu PT_state must have length 6 (1D) or 9 (2D); got {len(PT_state)}"
            )
        T = gauss_conv(T_rough, sigma=3, axis=0, mode="nearest")
    elif PT_profile == "slope":
        T_phot = PT_state[0]
        Delta_T_arr = np.array(PT_state[1:])
        T_rough = compute_T_slope(
            P, T_phot, Delta_T_arr, log_P_slope_phot, log_P_slope_arr
        )
        smooth_width = round(0.3 / ((np.log10(P[0]) - np.log10(P[-1])) / len(P)))
        T = gauss_conv(T_rough, sigma=smooth_width, axis=0, mode="nearest")
    elif PT_profile == "Pelletier":
        T_points = PT_state[:-1] if PT_penalty else PT_state
        T = compute_T_Pelletier(P, T_points)
    elif PT_profile == "Guillot":
        log_kappa_IR, log_gamma, T_int, T_equ = PT_state
        T = compute_T_Guillot(P, g_0, log_kappa_IR, log_gamma, T_int, T_equ)
    elif PT_profile == "Guillot_dayside":
        log_kappa_IR, log_gamma, T_int, T_equ = PT_state
        T = compute_T_Guillot_dayside(P, g_0, log_kappa_IR, log_gamma, T_int, T_equ)
    elif PT_profile == "Line":
        log_kappa_IR, log_gamma, log_gamma_2, alpha_l, beta_l, T_int = PT_state
        if T_eq is None:
            raise ValueError("PT_profile='Line' requires T_eq")
        T = compute_T_Line(
            P, g_0, T_eq, log_kappa_IR, log_gamma, log_gamma_2, alpha_l, beta_l, T_int
        )
    else:  # PT_profile == "file_read"
        if T_input is None:
            raise ValueError("PT_profile='file_read' requires T_input")
        T = T_input.reshape((len(P), 1, 1))

    N_species = len(included_species)

    species_has_profile = np.zeros(len(param_species)).astype(np.int64)
    if X_profile in ("gradient", "two-gradients"):
        species_has_profile[np.isin(param_species, species_vert_gradient)] = 1
    elif X_profile == "dissociation":
        if len(species_vert_gradient) == 0:
            species_has_profile[:] = 1
        else:
            species_has_profile[np.isin(param_species, species_vert_gradient)] = 1

    if X_profile == "file_read":
        if X_input is None:
            raise ValueError("X_profile='file_read' requires X_input")
        X = X_input.reshape((N_species, len(P), 1, 1))
    else:
        if X_profile in ("isochem", "gradient"):
            X_param = compute_X_field_gradient(
                P,
                log_X_state,
                N_sectors,
                N_zones,
                param_species,
                species_has_profile,
                alpha,
                beta,
                phi,
                theta,
            )
        elif X_profile == "two-gradients":
            X_param = compute_X_field_two_gradients(
                P,
                log_X_state,
                N_sectors,
                N_zones,
                param_species,
                species_has_profile,
                alpha,
                beta,
                phi,
                theta,
            )
        elif X_profile == "lever":
            if N_sectors != 1 or N_zones != 1:
                raise Exception(
                    "Error: Lever profile currently only implemented for "
                    "1D atmospheres."
                )
            X_param = compute_X_lever(
                P, log_X_state, species_has_profile, N_sectors, N_zones
            )
        elif X_profile == "dissociation":
            X_param = compute_X_dissociation(
                P,
                T,
                log_X_state,
                N_sectors,
                N_zones,
                param_species,
                species_has_profile,
                alpha,
                beta,
                phi,
                theta,
            )
        elif X_profile == "chem_eq":
            from jaxposeidon._chemistry import interpolate_log_X_grid

            C_to_O = log_X_state[0]
            log_Met = log_X_state[1]
            log_X_input = interpolate_log_X_grid(
                chemistry_grid,
                np.log10(P),
                T,
                C_to_O,
                log_Met,
                param_species,
                return_dict=False,
            )
            X_param = 10**log_X_input

        for q in range(len(param_species)):
            if species_has_profile[q] == 1:
                X_param[q, :, :, :] = gauss_conv(
                    X_param[q, :, :, :], sigma=3, axis=0, mode="nearest"
                )

        if Na_K_fixed_ratio:
            K_X_state = [X_param[list(param_species).index("Na")] * 0.1]
            X_param = np.append(X_param, K_X_state, axis=0)

        X = add_bulk_component(
            P, T, X_param, N_species, N_sectors, N_zones, bulk_species, He_fraction
        )

    if np.any(X[0, :, :, :] < 0.0):
        return (0,) * 12 + (False,)

    X_active, X_CIA, X_ff, X_bf = mixing_ratio_categories(
        P,
        X,
        N_sectors,
        N_zones,
        included_species,
        active_species,
        CIA_pairs,
        ff_pairs,
        bf_species,
    )

    masses_all = np.zeros(N_species)
    for q in range(N_species):
        sp = included_species[q]
        if sp == "ghost":
            if mu_back is None:
                raise ValueError("ghost bulk species requires explicit mu_back kwarg")
            masses_all[q] = mu_back
        else:
            masses_all[q] = _masses[sp]
    mu = compute_mean_mol_mass(P, X, N_species, N_sectors, N_zones, masses_all)

    if constant_gravity:
        n, r, r_up, r_low, dr = radial_profiles_constant_g(
            P,
            T,
            g_0,
            P_ref,
            R_p_ref,
            mu,
            N_sectors,
            N_zones,
        )
    else:
        n, r, r_up, r_low, dr = radial_profiles(
            P,
            T,
            g_0,
            R_p,
            P_ref,
            R_p_ref,
            mu,
            N_sectors,
            N_zones,
        )

    if np.any(r < 0.0):
        return (0,) * 12 + (False,)

    return T, n, r, r_up, r_low, dr, mu, X, X_active, X_CIA, X_ff, X_bf, True
