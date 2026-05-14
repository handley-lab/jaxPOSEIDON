"""Aerosol database loader (setup-only).

Setup-only: h5py / numpy / file I/O permitted. Must not be called from
inside `jit`. Allow-listed by the v1 source-grep gate.

Ports POSEIDON `clouds.py:1461-1639` (`load_aerosol_grid`) for the
default `aerosol` grid + log_r_m_std_dev = 0.5 case. The real grid
lives on Zenodo (DOI 10.5281/zenodo.15711943) and is loaded from
`$POSEIDON_input_data/opacity/aerosol_database.hdf5`. CI uses
synthetic HDF5 fixtures; env-gated real-grid smoke tests load the
shipped database.

The free-logwidth aerosol grid (`lognormal_logwith_free=True`) and the
non-default grids `SiO2_free_logwidth`, `aerosol_directional`,
`aerosol_diamonds` are deferred to a follow-up.
"""

import os

import h5py
import numpy as np

# POSEIDON's full supported list also includes 'SiO2_free_logwidth',
# 'aerosol_directional', 'aerosol_diamonds'; this port currently
# implements only the default 'aerosol' grid with fixed log_r_m_std_dev=0.5.
SUPPORTED_AEROSOL_GRIDS = frozenset({"aerosol"})


def load_aerosol_grid(aerosol_species, grid="aerosol"):
    """Load an aerosol cross-section grid from `$POSEIDON_input_data`.

    Bit-equivalent port of POSEIDON `clouds.py:1461-1639` for the
    default lognormal-width fixed-at-0.5 case. Returns a dict
    interchangeable with POSEIDON's:
        {'grid', 'sigma_Mie_grid', 'wl_grid', 'r_m_grid'}
    where `sigma_Mie_grid` has shape `(N_species, 3, r_m_num, wl_num)`
    storing `[eff_ext, eff_g, eff_w]` per species/radius/wavelength.
    """
    if grid not in SUPPORTED_AEROSOL_GRIDS:
        raise Exception("Error: unsupported aerosol grid")

    input_file_path = os.environ.get("POSEIDON_input_data")  # noqa: SIM112
    if input_file_path is None:
        raise Exception(
            "POSEIDON cannot locate the input folder.\n"
            "Please set the 'POSEIDON_input_data' variable in your .bashrc "
            "or .bash_profile to point to the POSEIDON input folder."
        )

    aerosol_species = np.array(aerosol_species)
    db_path = os.path.join(input_file_path, "opacity", f"{grid}_database.hdf5")
    try:
        database = h5py.File(db_path, "r")
    except Exception as exc:
        raise Exception(
            f"POSEIDON could not find {db_path}. Is it downloaded and in the "
            "opacity folder?"
        ) from exc

    wl_grid = np.array(database["Info/Wavelength grid"])
    r_m_grid = np.array(database["Info/Particle Size grid"])

    wl_num = len(wl_grid)
    r_m_num = len(r_m_grid)
    N_species = len(aerosol_species)
    sigma_Mie_grid = np.zeros((N_species, 3, r_m_num, wl_num))

    for q, species in enumerate(aerosol_species):
        ext_array = np.array(database[species]["0.5"]["eff_ext"]).reshape(
            r_m_num, wl_num
        )
        g_array = np.array(database[species]["0.5"]["eff_g"]).reshape(r_m_num, wl_num)
        w_array = np.array(database[species]["0.5"]["eff_w"]).reshape(r_m_num, wl_num)
        sigma_Mie_grid[q, 0, :, :] = ext_array
        sigma_Mie_grid[q, 1, :, :] = g_array
        sigma_Mie_grid[q, 2, :, :] = w_array

    database.close()
    return {
        "grid": grid,
        "sigma_Mie_grid": sigma_Mie_grid,
        "wl_grid": wl_grid,
        "r_m_grid": r_m_grid,
    }
