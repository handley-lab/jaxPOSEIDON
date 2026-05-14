"""Chemistry runtime — equilibrium-chemistry grid interpolation.

Ports POSEIDON `chemistry.py:119-271` (`interpolate_log_X_grid`). The
grid loader lives in `_fastchem_grid_loader.py` (setup-only, file I/O).
Runtime interpolation runs in this module via
`scipy.interpolate.RegularGridInterpolator` (matched to POSEIDON's
exact primitive).

`load_chemistry_grid` is re-exported here from
`_fastchem_grid_loader` for callers that expect POSEIDON's
`from POSEIDON.chemistry import load_chemistry_grid` pattern.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jaxposeidon._fastchem_grid_loader import (  # noqa: F401
    load_chemistry_grid,
)
from jaxposeidon._species_data import fastchem_supported_species


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

    len_P = np.array(log_P).size
    len_T = np.array(T).size
    len_C_to_O = np.array(C_to_O).size
    len_Met = np.array(log_Met).size
    max_len = max(len_P, len_T, len_C_to_O, len_Met)

    np.seterr(divide="ignore")

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

    def not_valid(params, grid, is_log):
        if is_log:
            return (10 ** np.max(params) < grid[0]) or (10 ** np.min(params) > grid[-1])
        else:
            return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(log_P, P_grid, True):
        raise Exception("Requested pressure is out of the grid bounds.")
    if not_valid(T, T_grid, False):
        raise Exception("Requested temperature is out of the grid bounds.")
    if not_valid(C_to_O, C_to_O_grid, False):
        raise Exception("Requested C/O is out of the grid bounds.")
    if not_valid(log_Met, Met_grid, True):
        raise Exception("Requested M/H is out of the grid bounds.")

    T = np.asarray(T)
    if T.ndim == 3:
        T_shape = T.shape
        assert len_C_to_O == 1
        assert len_Met == 1
        log_P = np.asarray(log_P)
        assert len(log_P.shape) == 1
        assert log_P.shape[0] == T_shape[0]

        reps = np.array(T_shape[1:])
        reps = np.insert(reps, 0, 1)
        log_P = log_P.reshape(-1, 1, 1)
        log_P = np.tile(log_P, reps)
        C_to_O = np.full(T_shape, C_to_O)
        log_Met = np.full(T_shape, log_Met)
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

        if len_P == 1:
            log_P = np.full(max_len, log_P)
        if len_T == 1:
            T = np.full(max_len, T)
        if len_C_to_O == 1:
            C_to_O = np.full(max_len, C_to_O)
        if len_Met == 1:
            log_Met = np.full(max_len, log_Met)

    def interpolate(species):
        q = np.where(np.asarray(chemical_species) == species)[0][0]
        grid_interp = RegularGridInterpolator(
            (np.log10(Met_grid), C_to_O_grid, T_grid, np.log10(P_grid)),
            log_X_grid[q, :, :, :, :],
        )
        return grid_interp(
            np.vstack(
                (
                    np.expand_dims(log_Met, 0),
                    np.expand_dims(C_to_O, 0),
                    np.expand_dims(T, 0),
                    np.expand_dims(log_P, 0),
                )
            ).T
        ).T

    if not return_dict:
        if isinstance(chemical_species, str):
            return interpolate(chemical_species)
        log_X_list = []
        for species in chemical_species:
            log_X_list.append(interpolate(species))
        return np.array(log_X_list)
    else:
        log_X_interp_dict = {}
        if isinstance(chemical_species, str):
            log_X_interp_dict[chemical_species] = interpolate(chemical_species)
            return log_X_interp_dict
        for species in chemical_species:
            log_X_interp_dict[species] = interpolate(species)
        return log_X_interp_dict
