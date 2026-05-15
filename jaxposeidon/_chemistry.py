"""Chemistry runtime — equilibrium-chemistry grid interpolation.

Ports POSEIDON `chemistry.py:119-271` (`interpolate_log_X_grid`). The
grid loader lives in `_fastchem_grid_loader.py` (setup-only, file I/O).

v1-B: numeric kernel ported to `jax.numpy` via the v1-A
`regular_grid_interp_linear` primitive; shape munging stays in
plain Python (input shape dispatch is decided at trace time from
static shapes).

`load_chemistry_grid` is re-exported here from
`_fastchem_grid_loader` for callers that expect POSEIDON's
`from POSEIDON.chemistry import load_chemistry_grid` pattern.

Documented divergence (see `MISMATCHES.md`): POSEIDON's string-valued
`chemical_species` path crashes on numpy >= 2.x; the port handles it
explicitly while preserving the iterable behaviour bit-exactly.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from jaxposeidon._fastchem_grid_loader import (  # noqa: E402, F401
    load_chemistry_grid,
)
from jaxposeidon._jax_interpolate import (  # noqa: E402
    regular_grid_interp_linear,
)
from jaxposeidon._species_data import fastchem_supported_species  # noqa: E402


def interpolate_log_X_grid(
    chemistry_grid, log_P, T, C_to_O, log_Met, chemical_species, return_dict=True
):
    """Interpolate FastChem log-mixing-ratio grid onto a P-T-Met-C/O profile.

    Bit-equivalent port of POSEIDON `chemistry.py:119-271`. Accepts
    POSEIDON's 3D-T shape convention plus the scalar/1D fan-out cases.
    """
    grid = chemistry_grid["grid"]
    log_X_grid = chemistry_grid["log_X_grid"]
    T_grid = chemistry_grid["T_grid"]
    P_grid = chemistry_grid["P_grid"]
    Met_grid = chemistry_grid["Met_grid"]
    C_to_O_grid = chemistry_grid["C_to_O_grid"]

    def _size(x):
        if hasattr(x, "shape"):
            return int(np.prod(x.shape)) if x.shape else 1
        return np.asarray(x).size

    def _ndim(x):
        if hasattr(x, "shape"):
            return len(x.shape)
        return np.asarray(x).ndim

    def _shape(x):
        if hasattr(x, "shape"):
            return tuple(x.shape)
        return np.asarray(x).shape

    len_P = _size(log_P)
    len_T = _size(T)
    len_C_to_O = _size(C_to_O)
    len_Met = _size(log_Met)
    max_len = max(len_P, len_T, len_C_to_O, len_Met)

    # Bounds checks (Python-side; skipped when called under jit on a
    # caller that has already clipped the inputs to the grid).
    if not hasattr(log_P, "aval"):

        def not_valid(params, grid, is_log):
            params = np.asarray(params)
            if is_log:
                return (10 ** np.max(params) < grid[0]) or (
                    10 ** np.min(params) > grid[-1]
                )
            return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

        if not_valid(log_P, P_grid, True):
            raise Exception("Requested pressure is out of the grid bounds.")
        if not_valid(T, T_grid, False):
            raise Exception("Requested temperature is out of the grid bounds.")
        if not_valid(C_to_O, C_to_O_grid, False):
            raise Exception("Requested C/O is out of the grid bounds.")
        if not_valid(log_Met, Met_grid, True):
            raise Exception("Requested M/H is out of the grid bounds.")

    if grid == "fastchem":
        supported_species = fastchem_supported_species
    else:
        raise Exception("Error: unsupported chemistry grid")
    if isinstance(chemical_species, str):
        if chemical_species not in supported_species:
            raise Exception(
                f"{chemical_species} is not supported by the equilibrium grid."
            )
    else:
        for species in chemical_species:
            if species not in supported_species:
                raise Exception(f"{species} is not supported by the equilibrium grid.")

    T_arr = jnp.asarray(T)
    if _ndim(T) == 3:
        T_shape = _shape(T)
        assert len_C_to_O == 1
        assert len_Met == 1
        log_P_arr = jnp.asarray(log_P)
        assert log_P_arr.ndim == 1
        assert log_P_arr.shape[0] == T_shape[0]

        log_P_b = jnp.broadcast_to(log_P_arr[:, None, None], T_shape)
        C_to_O_scalar = jnp.asarray(C_to_O).reshape(())
        log_Met_scalar = jnp.asarray(log_Met).reshape(())
        C_to_O_b = jnp.broadcast_to(C_to_O_scalar, T_shape)
        log_Met_b = jnp.broadcast_to(log_Met_scalar, T_shape)
        log_P_q = log_P_b
        T_q = T_arr
        C_to_O_q = C_to_O_b
        log_Met_q = log_Met_b
    else:
        if not (
            len_P in (1, max_len)
            and len_T in (1, max_len)
            and len_C_to_O in (1, max_len)
            and len_Met in (1, max_len)
        ):
            raise Exception(
                "Input shape not accepted. The lengths must either be the same or 1."
            )

        def _broadcast(x, n):
            x_arr = jnp.asarray(x).reshape((-1,))
            if x_arr.shape[0] == n:
                return x_arr
            return jnp.broadcast_to(x_arr.reshape(()), (n,))

        log_P_q = _broadcast(log_P, max_len)
        T_q = _broadcast(T, max_len)
        C_to_O_q = _broadcast(C_to_O, max_len)
        log_Met_q = _broadcast(log_Met, max_len)

    axes = (
        np.log10(np.asarray(Met_grid)),
        np.asarray(C_to_O_grid),
        np.asarray(T_grid),
        np.log10(np.asarray(P_grid)),
    )

    def interpolate(species):
        if isinstance(chemical_species, str):
            if species != chemical_species:
                raise KeyError(species)
            q = 0
        else:
            q = int(np.where(np.asarray(chemical_species) == species)[0][0])
        values = jnp.asarray(log_X_grid[q, :, :, :, :], dtype=jnp.float64)
        qp = jnp.stack([log_Met_q, C_to_O_q, T_q, log_P_q], axis=-1)
        return regular_grid_interp_linear(axes, values, qp)

    if not return_dict:
        if isinstance(chemical_species, str):
            return interpolate(chemical_species)
        log_X_list = [interpolate(s) for s in chemical_species]
        return jnp.stack(log_X_list, axis=0)
    else:
        log_X_interp_dict = {}
        if isinstance(chemical_species, str):
            log_X_interp_dict[chemical_species] = interpolate(chemical_species)
            return log_X_interp_dict
        for species in chemical_species:
            log_X_interp_dict[species] = interpolate(species)
        return log_X_interp_dict
