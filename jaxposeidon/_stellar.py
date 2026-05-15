"""Stellar contamination forward model (v1-D JAX port).

Ports POSEIDON `stellar.py:26-794` numerical functions:
- planck_lambda                  — single-T black-body radiance
- stellar_contamination_single_spot — Rackham+17/18 single-spot factor
- stellar_contamination_general — multi-region collective factor

The pysynphot / PyMSG grid loaders live in `_stellar_grid_loader.py`
(setup-only). The runtime `stellar_contamination` dispatch from a star
dict is provided here as a thin JAX-traceable wrapper that supports
'one_spot', 'two_spots', and 'three_spots' (general multi-region path).
"""

import jax
import jax.numpy as jnp
import scipy.constants as sc

jax.config.update("jax_enable_x64", True)

_H = float(sc.h)
_C = float(sc.c)
_K = float(sc.k)


def planck_lambda(T, wl):
    """Single-temperature black-body radiance B(T, λ) in SI (W/m²/sr/m).

    Bit-equivalent port of POSEIDON `stellar.py:26-60`.
    """
    T = jnp.asarray(T, dtype=jnp.float64)
    wl = jnp.asarray(wl, dtype=jnp.float64)
    wl_m = wl * 1.0e-6
    c_2 = (_H * _C) / _K
    coeff = (2.0 * _H * _C**2) / (wl_m**5)
    return coeff / (jnp.exp(c_2 / (wl_m * T)) - 1.0)


def stellar_contamination_single_spot(f, I_het, I_phot):
    """Rackham+17/18 single-region stellar contamination factor.

    Bit-equivalent port of POSEIDON `stellar.py:733-756`.
    `ε(λ) = 1 / (1 - f · (1 - I_het / I_phot))`
    """
    f = jnp.asarray(f, dtype=jnp.float64)
    I_het = jnp.asarray(I_het, dtype=jnp.float64)
    I_phot = jnp.asarray(I_phot, dtype=jnp.float64)
    return 1.0 / (1.0 - f * (1.0 - I_het / I_phot))


def stellar_contamination_general(f_het, I_het, I_phot):
    """Multi-region Rackham+17/18 stellar contamination factor.

    Bit-equivalent port of POSEIDON `stellar.py:760-794`. Sums each
    region's `f_i · (1 - I_het,i / I_phot)` then applies the same
    `1 / (1 - Σ)` form.
    """
    f_het = jnp.asarray(f_het, dtype=jnp.float64)
    I_het = jnp.asarray(I_het, dtype=jnp.float64)
    I_phot = jnp.asarray(I_phot, dtype=jnp.float64)
    I_ratio = I_het / I_phot[None, :]
    total = jnp.sum(f_het[:, None] * (1.0 - I_ratio), axis=0)
    return 1.0 / (1.0 - total)


def apply_stellar_contamination(spectrum, star, stellar_params):
    """Apply stellar contamination to a transmission/emission spectrum.

    Mirrors POSEIDON `stellar.py:797-863` for the wavelength-matched
    case (no spectres interpolation). ``stellar_params`` (sampled at
    retrieval time) overrides ``star['f_het']`` / ``star['I_het']`` if
    its layout matches: for ``one_spot`` the first element is treated
    as the active ``f_het``; for ``two_spots`` / ``three_spots`` the
    leading ``N`` elements are filling factors. POSEIDON's full
    parameter ordering (T_phot, T_het, log_g_het, …) involves
    intensity-grid lookup that requires pysynphot / PyMSG and is
    deferred to a follow-up.
    """
    if star is None:
        return spectrum
    contam = star.get("stellar_contam")
    if contam is None:
        return spectrum
    spectrum = jnp.asarray(spectrum, dtype=jnp.float64)
    I_phot = jnp.asarray(star["I_phot"], dtype=jnp.float64)
    stellar_params = jnp.asarray(stellar_params, dtype=jnp.float64)
    has_params = stellar_params.shape[0] > 0
    if contam == "one_spot":
        if has_params:
            f = stellar_params[0]
        else:
            f = jnp.asarray(star["f_het"], dtype=jnp.float64)
        I_het = jnp.asarray(star["I_het"], dtype=jnp.float64)
        eps = stellar_contamination_single_spot(f, I_het, I_phot)
        return spectrum * eps
    if contam in ("two_spots", "three_spots"):
        n_het = 2 if contam == "two_spots" else 3
        if has_params:
            f_het = stellar_params[:n_het]
        else:
            f_het = jnp.asarray(star["f_het"], dtype=jnp.float64)
        I_het = jnp.asarray(star["I_het"], dtype=jnp.float64)
        eps = stellar_contamination_general(f_het, I_het, I_phot)
        return spectrum * eps
    raise NotImplementedError(f"stellar_contam={contam!r} not supported")
