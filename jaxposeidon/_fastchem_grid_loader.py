"""FastChem grid loader (setup-only).

Setup-only: h5py / numpy / file I/O permitted. Must not be called from
inside `jit`. Allow-listed by the v1 source-grep gate.

Ports POSEIDON `chemistry.py:16-116` (`load_chemistry_grid`). The
FastChem equilibrium-chemistry grid is shipped via POSEIDON's
`POSEIDON_input_data/chemistry_grids/fastchem_database.hdf5` (~1 GB
when populated). CI uses synthetic fixtures; the real grid is loaded
via env-gated smoke tests.
"""

import os

import h5py
import numpy as np

from jaxposeidon._species_data import fastchem_supported_species


def load_chemistry_grid(chemical_species, grid="fastchem"):
    """Load a FastChem chemistry grid from `$POSEIDON_input_data`.

    Bit-equivalent port of POSEIDON `chemistry.py:16-116`. Drops the
    POSEIDON MPI shared-memory allocation (single-process), but the
    returned dict is interchangeable with POSEIDON's.
    """
    if grid not in ("fastchem",):
        raise Exception("Error: unsupported chemistry grid")

    input_file_path = os.environ.get("POSEIDON_input_data")  # noqa: SIM112
    if input_file_path is None:
        raise Exception(
            "POSEIDON cannot locate the input folder.\n"
            "Please set the 'POSEIDON_input_data' variable in your .bashrc "
            "or .bash_profile to point to the POSEIDON input folder."
        )

    supported_chem_eq_species = fastchem_supported_species
    if "all" in chemical_species:
        chemical_species = np.array(supported_chem_eq_species)
    else:
        if np.any(
            ~np.isin(np.array(chemical_species), np.array(supported_chem_eq_species))
        ):
            raise Exception(
                "A chemical species you selected is not supported "
                "for equilibrium chemistry models.\n"
            )

    db_path = os.path.join(input_file_path, "chemistry_grids", f"{grid}_database.hdf5")
    database = h5py.File(db_path, "r")
    T_grid = np.array(database["Info/T grid"])
    P_grid = np.array(database["Info/P grid"])
    Met_grid = np.array(database["Info/M/H grid"])
    C_to_O_grid = np.array(database["Info/C/O grid"])

    T_num = len(T_grid)
    P_num = len(P_grid)
    Met_num = len(Met_grid)
    C_O_num = len(C_to_O_grid)

    N_species = len(chemical_species)
    log_X_grid = np.zeros((N_species, Met_num, C_O_num, T_num, P_num))
    for q, species in enumerate(chemical_species):
        array = np.array(database[f"{species}/log(X)"])
        array = array.reshape(Met_num, C_O_num, T_num, P_num)
        log_X_grid[q, :, :, :, :] = array
    database.close()

    return {
        "grid": grid,
        "log_X_grid": log_X_grid,
        "T_grid": T_grid,
        "P_grid": P_grid,
        "Met_grid": Met_grid,
        "C_to_O_grid": C_to_O_grid,
    }
