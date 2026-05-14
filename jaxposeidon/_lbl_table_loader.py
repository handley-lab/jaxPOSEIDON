"""LBL opacity table loader (setup-only).

Setup-only: h5py / file I/O permitted. Must not be called from inside
`jit`. Allow-listed by the v1 source-grep gate.

The LBL HDF5 tables (`Opacity_database_v1.3.hdf5` ~10+ GB,
`Opacity_database_cia.hdf5` ~few GB) live at
`$POSEIDON_input_data/opacity/`. The loader signatures mirror
POSEIDON's `absorption.py:1739-1951` file-opening logic so callers can
inject the same opened h5py.File handles consumed by `compute_kappa_LBL`.

The full extinction_LBL orchestrator is a follow-up; this slice gives
a stable place for the path-construction logic to live.
"""

import os

import h5py

SUPPORTED_OPACITY_DATABASES = frozenset({"High-T", "Temperate"})


def open_opacity_files(opacity_database="High-T", database_version="1.3"):
    """Open POSEIDON's molecular + CIA HDF5 opacity files.

    Bit-equivalent path-construction port of POSEIDON
    `absorption.py:1794-1818`. Returns `(opac_file, cia_file)` open
    h5py.File handles. Caller is responsible for closing them.
    """
    if opacity_database not in SUPPORTED_OPACITY_DATABASES:
        raise Exception(
            f"Unsupported opacity database {opacity_database!r}; "
            f"must be one of {sorted(SUPPORTED_OPACITY_DATABASES)}."
        )
    input_file_path = os.environ["POSEIDON_input_data"]  # noqa: SIM112

    if opacity_database == "High-T":
        if database_version == "1.3":
            opac_path = os.path.join(
                input_file_path, "opacity", "Opacity_database_v1.3.hdf5"
            )
        elif database_version == "1.2":
            opac_path = os.path.join(
                input_file_path, "opacity", "Opacity_database_v1.2.hdf5"
            )
        elif database_version == "1.0":
            opac_path = os.path.join(
                input_file_path, "opacity", "Opacity_database_v1.0.hdf5"
            )
        else:
            raise Exception(
                "Invalid opacity database version.\n"
                "The options are: '1.0', '1.2', or '1.3."
            )
    else:  # Temperate
        opac_path = os.path.join(
            input_file_path, "opacity", "Opacity_database_0.01cm-1_Temperate.hdf5"
        )

    cia_path = os.path.join(input_file_path, "opacity", "Opacity_database_cia.hdf5")
    opac_file = h5py.File(opac_path, "r")
    cia_file = h5py.File(cia_path, "r")
    return opac_file, cia_file
