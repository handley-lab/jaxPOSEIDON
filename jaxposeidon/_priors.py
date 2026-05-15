"""Unit-cube prior transform — v1 JAX port.

Faithful port of POSEIDON `retrieval.py:547-1008`. Hot-path numeric
kernels use `jnp`; per-parameter dispatch is encoded as integer codes
so the dispatcher can run under `jax.jit` (the equivalent Python-string
dispatch in v0.5 is preserved for the existing public API).

Prior types and POSEIDON oracle references:
- 'uniform'   (`retrieval.py:651-657`)
- 'gaussian'  (`retrieval.py:658-665`)  uses `scipy.special.ndtri`
- 'sine' (alpha/beta)   (`retrieval.py:666-678`)
- 'sine' (theta_0)      (`retrieval.py:679-684`)
- 'CLR' (mixing ratios) (`retrieval.py:547-594, 861-887`)
"""

from functools import partial

import jax.numpy as jnp
import numpy as np
from jax import jit

from jaxposeidon._jax_special import ndtri

PRIOR_TYPES = frozenset({"uniform", "gaussian", "sine", "CLR"})
SINE_ALPHA_BETA = frozenset({"alpha", "beta"})
SINE_THETA_0 = "theta_0"

_PTYPE_UNIFORM = 0
_PTYPE_GAUSSIAN = 1
_PTYPE_SINE_ALPHA_BETA = 2
_PTYPE_SINE_THETA_0 = 3
_PTYPE_CLR = 4


def _ptype_code(name, ptype):
    if ptype == "uniform":
        return _PTYPE_UNIFORM
    if ptype == "gaussian":
        return _PTYPE_GAUSSIAN
    if ptype == "sine":
        if name in SINE_ALPHA_BETA:
            return _PTYPE_SINE_ALPHA_BETA
        if name == SINE_THETA_0:
            return _PTYPE_SINE_THETA_0
        raise NotImplementedError(
            f"sine prior on {name!r} not in v0 "
            "(POSEIDON only uses sine for alpha/beta/theta_0)"
        )
    if ptype == "CLR":
        return _PTYPE_CLR
    raise NotImplementedError(
        f"prior_type={ptype!r} not supported (uniform/gaussian/sine/CLR)"
    )


def CLR_Prior(chem_params_drawn, limit=-12.0):
    """Centred-log-ratio prior for chemical mixing ratios.

    Bit-equivalent port of POSEIDON `retrieval.py:547-594`. Returns a
    length-(n+1) array of log10 mixing ratios on success, or
    `[-50.0]*(n+1)` if the draw falls outside the allowed simplex.

    Implementation uses `jnp` arithmetic with `jnp.where`-based
    branching so the kernel is jit-able. The Python ``len(...)`` call
    on the input keeps `n` static; pass a fixed-size array under jit.
    """
    chem_params_drawn = jnp.asarray(chem_params_drawn, dtype=jnp.float64)
    n = chem_params_drawn.shape[0]

    prior_lower_CLR = (((n + 1) - 1.0) / (n + 1)) * (
        limit * jnp.log(10.0) + jnp.log((n + 1) - 1.0)
    )
    prior_upper_CLR = ((1.0 - (n + 1)) / (n + 1)) * (limit * jnp.log(10.0))

    CLR_tail = chem_params_drawn * (prior_upper_CLR - prior_lower_CLR) + prior_lower_CLR
    CLR_head = -jnp.sum(CLR_tail)
    CLR = jnp.concatenate([CLR_head[None], CLR_tail])

    normalisation = jnp.sum(jnp.exp(CLR))
    X = jnp.exp(CLR) / normalisation
    log_X = jnp.log10(X)

    sentinel = jnp.full((n + 1,), -50.0, dtype=jnp.float64)

    cond_sum = jnp.abs(jnp.sum(CLR_tail)) <= prior_upper_CLR
    cond_range = (jnp.max(CLR) - jnp.min(CLR)) <= (-limit * jnp.log(10.0))
    cond_min_X = jnp.min(X) >= 1.0e-12

    accept = cond_sum & cond_range & cond_min_X
    return jnp.where(accept, log_X, sentinel)


def prior_transform(
    unit_cube,
    param_names,
    prior_types,
    prior_ranges,
    X_param_names=None,
    N_params_cum=None,
):
    """Map a unit hypercube vector to physical parameters.

    Mirrors POSEIDON `retrieval.py:649-887`. When any mixing-ratio
    parameter uses prior_type='CLR', `X_param_names` and `N_params_cum`
    (the cumulative-parameter boundary array from `assign_free_params`)
    must be supplied. Rejected CLR draws fill the chemistry slice with
    the sentinel value -50.0 (POSEIDON's allowed_simplex convention,
    retrieval.py:861-887); downstream likelihood checks for that.

    Public signature is preserved from v0.5 so the existing test surface
    keeps working. For jit-able callers, see
    `prior_transform_kernel(unit_cube, codes, range_lo, range_hi, ...)`.
    """
    for parameter in param_names:
        if prior_types[parameter] not in PRIOR_TYPES:
            raise NotImplementedError(
                f"prior_type={prior_types[parameter]!r} not supported "
                "(uniform/gaussian/sine/CLR)"
            )

    if "CLR" in prior_types.values() and (
        X_param_names is None or N_params_cum is None
    ):
        raise ValueError("CLR prior requires X_param_names and N_params_cum kwargs")

    codes = np.array(
        [_ptype_code(p, prior_types[p]) for p in param_names], dtype=np.int32
    )
    range_lo = np.array([prior_ranges[p][0] for p in param_names], dtype=np.float64)
    range_hi = np.array([prior_ranges[p][1] for p in param_names], dtype=np.float64)
    unit_cube_arr = jnp.asarray(unit_cube, dtype=jnp.float64)

    if X_param_names is not None and "CLR" in prior_types.values():
        N_species_params = len(X_param_names)
        CLR_lo = int(N_params_cum[1])
        CLR_hi = int(N_params_cum[2])
        CLR_limit = float(prior_ranges[X_param_names[0]][0])
        return _kernel_with_CLR(
            unit_cube_arr,
            jnp.asarray(codes),
            jnp.asarray(range_lo),
            jnp.asarray(range_hi),
            CLR_lo,
            CLR_hi,
            N_species_params,
            CLR_limit,
        )

    return _kernel_no_CLR(
        unit_cube_arr,
        jnp.asarray(codes),
        jnp.asarray(range_lo),
        jnp.asarray(range_hi),
    )


@jit
def _kernel_no_CLR(unit_cube, codes, range_lo, range_hi):
    return _apply_non_CLR(unit_cube, codes, range_lo, range_hi)


def _apply_non_CLR(unit_cube, codes, range_lo, range_hi):
    pi = jnp.float64(np.pi)
    uniform = unit_cube * (range_hi - range_lo) + range_lo
    gaussian = range_lo + range_hi * ndtri(unit_cube)
    sine_ab = (
        (180.0 / pi)
        * 2.0
        * jnp.arcsin(unit_cube * jnp.sin((pi / 180.0) * (range_hi / 2.0)))
    )
    sine_t0 = (180.0 / pi) * jnp.arcsin(
        (2.0 * unit_cube - 1.0) * jnp.sin((pi / 180.0) * (range_hi / 2.0))
    )

    out = unit_cube
    out = jnp.where(codes == _PTYPE_UNIFORM, uniform, out)
    out = jnp.where(codes == _PTYPE_GAUSSIAN, gaussian, out)
    out = jnp.where(codes == _PTYPE_SINE_ALPHA_BETA, sine_ab, out)
    out = jnp.where(codes == _PTYPE_SINE_THETA_0, sine_t0, out)
    return out


@partial(jit, static_argnames=("CLR_lo", "CLR_hi", "N_species_params", "CLR_limit"))
def _kernel_with_CLR(
    unit_cube,
    codes,
    range_lo,
    range_hi,
    CLR_lo,
    CLR_hi,
    N_species_params,
    CLR_limit,
):
    out = _apply_non_CLR(unit_cube, codes, range_lo, range_hi)
    chem_drawn = unit_cube[CLR_lo:CLR_hi]
    log_X = CLR_Prior(chem_drawn, CLR_limit)
    return out.at[CLR_lo : CLR_lo + N_species_params].set(log_X[1:])
