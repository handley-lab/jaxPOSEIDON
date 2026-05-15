"""JAX-pure special functions for the v1 hot path.

- `expn_2(x)`: port of `scipy.special.expn(2, x)` for x >= 0
  (used by `_atmosphere.py:327`). Implements
  E_2(x) = exp(-x) - x * E_1(x), with E_1 computed by a power series
  for x <= 1 and a Lentz continued fraction for x > 1
  (Abramowitz & Stegun 5.1.11 / 5.1.22; same branching as scipy's
  cephes `expn.c`).
- `ndtri`: re-export of `jax.scipy.special.ndtri` (Gaussian inverse CDF;
  used by `_priors.py:19`).
"""

import jax.numpy as jnp
from jax.scipy.special import ndtri  # noqa: F401  (re-exported)

EULER = 0.57721566490153286060651209008240243104215933593992

_SERIES_TERMS = 60
_CF_ITERS = 80


def _e1_series(x):
    """E_1(x) = -gamma - ln(x) + sum_{k>=1} (-1)^(k+1) x^k / (k k!)."""
    x_safe = jnp.where(x > 0.0, x, 1.0)
    s = jnp.zeros_like(x)
    term = jnp.ones_like(x)
    sign = 1.0
    for k in range(1, _SERIES_TERMS + 1):
        term = term * x_safe / k
        s = s + sign * term / k
        sign = -sign
    return -EULER - jnp.log(x_safe) + s


def _e1_contfrac(x):
    """Lentz continued fraction for E_1(x), x > 0.

    E_1(x) = exp(-x) * (1 / (x + 1 / (1 + 1 / (x + 2 / (1 + 2 / (...))))))
    """
    tiny = 1.0e-300
    x_safe = jnp.where(x > 0.0, x, 1.0)
    b = x_safe + 1.0
    c = jnp.full_like(x_safe, 1.0 / tiny)
    d = 1.0 / b
    h = d
    for i in range(1, _CF_ITERS + 1):
        a = -(i * i) * 1.0
        b = b + 2.0
        d = b + a * d
        d = jnp.where(jnp.abs(d) < tiny, tiny, d)
        c = b + a / c
        c = jnp.where(jnp.abs(c) < tiny, tiny, c)
        d = 1.0 / d
        delta = c * d
        h = h * delta
    return h * jnp.exp(-x_safe)


def _expn2_asymptotic(x):
    """Asymptotic series E_2(x) = exp(-x)/x * sum_{k>=0} (-1)^k (k+1)! / x^k.

    Valid for large x; avoids catastrophic cancellation in
    exp(-x) - x * E_1(x). The series is divergent, so we stop
    accumulating once the absolute term starts growing again
    (optimal truncation for an asymptotic series).
    """
    x_safe = jnp.where(x > 0.0, x, 1.0)
    inv = 1.0 / x_safe
    s = jnp.ones_like(x)
    term = jnp.ones_like(x)
    abs_term = jnp.ones_like(x)
    keep = jnp.ones_like(x, dtype=bool)
    sign = -1.0
    for k in range(1, 60):
        term_new = term * (k + 1) * inv
        abs_new = jnp.abs(term_new)
        growing = abs_new > abs_term
        keep = keep & ~growing
        s = s + jnp.where(keep, sign * term_new, 0.0)
        sign = -sign
        term = term_new
        abs_term = jnp.where(keep, abs_new, abs_term)
    return jnp.exp(-x_safe) * inv * s


def expn_2(x):
    """E_2(x) for x >= 0; parity to `scipy.special.expn(2, x)`.

    Three regimes (matching scipy/cephes branches):
    - x == 0: returns 1
    - 0 < x <= 1: E_2 = exp(-x) - x * E_1(x) with E_1 by power series
    - 1 < x <= 30: same but E_1 by Lentz continued fraction
    - x > 30: asymptotic series for E_2 directly (avoids cancellation)
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    e1_s = _e1_series(x)
    e1_cf = _e1_contfrac(x)
    e1 = jnp.where(x <= 1.0, e1_s, e1_cf)
    direct = jnp.exp(-x) - x * e1
    asymp = _expn2_asymptotic(x)
    out = jnp.where(x > 50.0, asymp, direct)
    return jnp.where(x == 0.0, jnp.float64(1.0), out)
