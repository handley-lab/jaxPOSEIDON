"""Opacity preprocessing — pressure / wavelength / temperature interpolation.

Faithful port of the v0 subset of POSEIDON `absorption.py:28-322`
(`P_interpolate_wl_initialise_sigma`, `wl_initialise_cia`,
`T_interpolation_init`, `T_interpolate_sigma`, `T_interpolate_cia`)
plus the grid-index helpers `closest_index`, `prior_index`,
`prior_index_V2` from `utility.py`.

v1-B: numeric kernels rewritten in `jax.numpy` so the runtime
extinction path is jit-traceable. Setup-time primitives
(`closest_index`, `prior_index`, `prior_index_V2`) operate on
Python scalars at setup; jit-able versions
(`closest_index_jax`, `prior_index_V2_jax`) are exposed for the
runtime hot path in `_jax_opacities.extinction_jax`.

H-minus and Rayleigh cross sections are deferred to v1 unless K2-18b
requires them (paper indicates negligible H-/Rayleigh contribution at
~250 K).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402  (setup-time only; jit-runtime uses jnp)


# ---------------------------------------------------------------------------
# Grid-index helpers (port of POSEIDON utility.py)
# ---------------------------------------------------------------------------
def prior_index(value, grid, start=0):
    """Index of the prior grid point for a non-uniform monotonic grid.

    Mirrors POSEIDON `utility.py:67-101`. Setup-time scalar function.
    """
    if value > grid[-1]:
        return len(grid) - 1
    if value < grid[0]:
        value = grid[0]
    if value > grid[-2]:
        value = grid[-2]
    index = start
    for i in range(len(grid) - start):
        if grid[i + start] > value:
            index = (i + start) - 1
            break
    return index


def prior_index_V2(value, grid_start, grid_end, N_grid):
    """Prior index in a uniformly spaced grid.

    Mirrors POSEIDON `utility.py:170-205`. Setup-time scalar function.
    """
    if value < grid_start:
        return 0
    if value > grid_end:
        return N_grid - 1
    return int((N_grid - 1) * ((value - grid_start) / (grid_end - grid_start)))


def closest_index(value, grid_start, grid_end, N_grid):
    """Closest index in a uniformly spaced grid.

    Mirrors POSEIDON `utility.py:209-247`. Setup-time scalar function.
    """
    if N_grid == 1:
        return 0
    if value < grid_start:
        return 0
    if value > grid_end:
        return N_grid - 1
    i = (N_grid - 1) * ((value - grid_start) / (grid_end - grid_start))
    if (i % 1) <= 0.5:
        return int(i)
    else:
        return int(i) + 1


def closest_index_jax(value, grid_start, grid_end, N_grid):
    """jit-traceable counterpart of `closest_index`.

    `value` may be a traced scalar; `grid_start`, `grid_end`, `N_grid`
    are static Python ints/floats.
    """
    if N_grid == 1:
        return jnp.int32(0)
    span = grid_end - grid_start
    i_float = (N_grid - 1) * ((value - grid_start) / span)
    i_int = jnp.floor(i_float).astype(jnp.int32)
    frac = i_float - i_int
    rounded = jnp.where(frac <= 0.5, i_int, i_int + 1)
    clipped = jnp.clip(rounded, 0, N_grid - 1)
    return clipped


def prior_index_V2_jax(value, grid_start, grid_end, N_grid):
    """jit-traceable counterpart of `prior_index_V2`."""
    span = grid_end - grid_start
    i_float = (N_grid - 1) * ((value - grid_start) / span)
    i_int = jnp.floor(i_float).astype(jnp.int32)
    return jnp.clip(i_int, 0, N_grid - 1)


# ---------------------------------------------------------------------------
# Pressure + wavelength preprocessing for molecular opacity
# ---------------------------------------------------------------------------
def P_interpolate_wl_initialise_sigma(
    N_P_fine,
    N_T,
    N_P,
    N_wl,
    log_sigma,
    x,
    nu_model,
    b1,
    b2,
    nu_opac,
    N_nu,
    wl_interp="sample",
):
    """Interpolate log10(σ)[log_P, T, ν] → σ[log_P_fine, T, λ_model].

    Bit-exact port of POSEIDON `absorption.py:28-134`. Setup-time
    function: produces opacity tables used at runtime.

    Numerics use plain numpy (numba-equivalent reduction order); jit is
    not required because this runs once per retrieval setup.
    """
    sigma_pre_inp = np.zeros((N_P_fine, N_T, N_wl))
    N_nu_opac = len(nu_opac)

    for k in range(N_nu):
        if wl_interp == "sample":
            z = closest_index(nu_model[k], nu_opac[0], nu_opac[-1], N_nu_opac)
            c1 = c2 = 0.0
        else:
            z = prior_index_V2(nu_model[k], nu_opac[0], nu_opac[-1], N_nu_opac)
            if (z == 0) or (z == (N_nu_opac - 1)):
                c1 = c2 = 0.0
            else:
                w_nu = (nu_model[k] - nu_opac[z]) / (nu_opac[z + 1] - nu_opac[z])
                c1 = 1.0 - w_nu
                c2 = w_nu

        for i in range(N_P_fine):
            for j in range(N_T):
                if (z == 0) or (z == (N_nu_opac - 1)):
                    sigma_pre_inp[i, j, (N_wl - 1) - k] = 0.0
                    continue

                if wl_interp == "sample":
                    if x[i] == -1:
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = 10 ** log_sigma[0, j, z]
                    elif x[i] == -2:
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = (
                            10 ** log_sigma[N_P - 1, j, z]
                        )
                    else:
                        reduced_sigma = log_sigma[x[i] : x[i] + 2, j, z]
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = 10 ** (
                            b1[i] * reduced_sigma[0] + b2[i] * reduced_sigma[1]
                        )
                else:
                    if x[i] == -1:
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = 10 ** (
                            c1 * log_sigma[0, j, z] + c2 * log_sigma[0, j, z + 1]
                        )
                    elif x[i] == -2:
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = 10 ** (
                            c1 * log_sigma[N_P - 1, j, z]
                            + c2 * log_sigma[N_P - 1, j, z + 1]
                        )
                    else:
                        log_sigma_rect = log_sigma[x[i] : x[i] + 2, j, z : z + 2]
                        log_sigma_nu_1 = (
                            b1[i] * log_sigma_rect[0, 0] + b2[i] * log_sigma_rect[1, 0]
                        )
                        log_sigma_nu_2 = (
                            b1[i] * log_sigma_rect[0, 1] + b2[i] * log_sigma_rect[1, 1]
                        )
                        sigma_pre_inp[i, j, (N_wl - 1) - k] = 10 ** (
                            c1 * log_sigma_nu_1 + c2 * log_sigma_nu_2
                        )

    return sigma_pre_inp


def wl_initialise_cia(
    N_T_cia, N_wl, log_cia, nu_model, nu_cia, N_nu, wl_interp="sample"
):
    """Interpolate log10(α_CIA)[T, ν] → α[T, λ_model].

    Bit-exact port of POSEIDON `absorption.py:138-200`. Setup-time
    function.
    """
    cia_pre_inp = np.zeros((N_T_cia, N_wl))
    N_nu_cia = len(nu_cia)

    for k in range(N_nu):
        if wl_interp == "sample":
            z = closest_index(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            c1 = c2 = 0.0
        else:
            z = prior_index_V2(nu_model[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            if (z == 0) or (z == (N_nu_cia - 1)):
                c1 = c2 = 0.0
            else:
                w_nu = (nu_model[k] - nu_cia[z]) / (nu_cia[z + 1] - nu_cia[z])
                c1 = 1.0 - w_nu
                c2 = w_nu

        for i in range(N_T_cia):
            if (z == 0) or (z == (N_nu_cia - 1)):
                cia_pre_inp[i, (N_wl - 1) - k] = 0.0
            else:
                if wl_interp == "sample":
                    cia_pre_inp[i, (N_wl - 1) - k] = 10 ** log_cia[i, z]
                else:
                    cia_pre_inp[i, (N_wl - 1) - k] = 10 ** (
                        c1 * log_cia[i, z] + c2 * log_cia[i, z + 1]
                    )

    return cia_pre_inp


# ---------------------------------------------------------------------------
# Temperature interpolation
# ---------------------------------------------------------------------------
def T_interpolation_init(N_T_fine, T_grid, T_fine, y):
    """Precompute T-interp weights and out-of-grid sentinels.

    Bit-exact port of POSEIDON `absorption.py:204-236`. Setup-time
    function; modifies `y` in place (out parameter), returns `w_T`.
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


def T_interpolate_sigma(
    N_P_fine, N_T_fine, N_T, N_wl, sigma_pre_inp, T_grid, T_fine, y, w_T
):
    """Interpolate σ from coarse T_grid onto T_fine (1/T-weighted geom mean).

    Bit-exact port of POSEIDON `absorption.py:240-279`. Setup-time
    function; produces opacity table consumed at runtime.

    `jnp`-pure version (`T_interpolate_sigma_jax`) is exposed for
    callers that need to differentiate through the T-interp; the
    default setup path stays in numpy for memory efficiency on the
    large grids typical of opacity-sampling mode.
    """
    sigma_inp = np.zeros((N_P_fine, N_T_fine, N_wl))
    for i in range(N_P_fine):
        for j in range(N_T_fine):
            T = T_fine[j]
            if y[j] == -1:
                sigma_inp[i, j, :] = sigma_pre_inp[i, 0, :]
            elif y[j] == -2:
                sigma_inp[i, j, :] = sigma_pre_inp[i, N_T - 1, :]
            else:
                T1 = T_grid[y[j]]
                T2 = T_grid[y[j] + 1]
                sig_1 = sigma_pre_inp[i, y[j], :]
                sig_2 = sigma_pre_inp[i, y[j] + 1, :]
                sigma_inp[i, j, :] = np.power(
                    sig_1, w_T[j] * ((1.0 / T2) - (1.0 / T))
                ) * np.power(sig_2, w_T[j] * ((1.0 / T) - (1.0 / T1)))
    return sigma_inp


def T_interpolate_cia(N_T_fine, N_T_cia, N_wl, cia_pre_inp, T_grid_cia, T_fine, y, w_T):
    """Interpolate CIA α from coarse T_grid_cia onto T_fine.

    Bit-exact port of POSEIDON `absorption.py:283-322`. Setup-time
    function.
    """
    cia_inp = np.zeros((N_T_fine, N_wl))
    for j in range(N_T_fine):
        T = T_fine[j]
        if y[j] == -1:
            cia_inp[j, :] = cia_pre_inp[0, :]
        elif y[j] == -2:
            cia_inp[j, :] = cia_pre_inp[N_T_cia - 1, :]
        else:
            T1 = T_grid_cia[y[j]]
            T2 = T_grid_cia[y[j] + 1]
            cia_1 = cia_pre_inp[y[j], :]
            cia_2 = cia_pre_inp[y[j] + 1, :]
            cia_inp[j, :] = np.power(
                cia_1, w_T[j] * ((1.0 / T2) - (1.0 / T))
            ) * np.power(cia_2, w_T[j] * ((1.0 / T) - (1.0 / T1)))
    return cia_inp
