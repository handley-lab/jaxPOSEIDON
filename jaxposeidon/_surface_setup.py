"""Surface setup-only module — parameter parsing + albedo file I/O.

Setup-only: numpy / pandas / file I/O permitted. Must not be called
from inside `jit`. Allow-listed by the v1 source-grep gate.

Ports POSEIDON `surfaces.py:1-100`:
- `find_nearest_less_than`: pure-numpy helper.
- `load_surface_components`: reads `surface_reflectivities/<name>.txt`
  from `$POSEIDON_input_data`. Each file has two columns (wavelength
  μm, albedo).
- `interpolate_surface_components`: linear interpolation of each
  loaded albedo onto the model wavelength grid.

The spectral effect of surfaces in emission/reflection/transmission
is deferred to Phase 0.5.13d. This module handles only the parser
and interpolator (no spectrum modification).
"""

import contextlib
import os

import numpy as np
from scipy.interpolate import interp1d


def find_nearest_less_than(searchVal, array):
    """Index of array element closest to but not exceeding searchVal.

    Bit-equivalent port of POSEIDON `surfaces.py:17-25`.
    """
    diff = array - searchVal
    with contextlib.suppress(Exception):
        diff[diff > 0] = -np.inf
    return diff.argmax()


def load_surface_components(surface_components):
    """Load albedo files from $POSEIDON_input_data/surface_reflectivities/.

    Bit-equivalent port of POSEIDON `surfaces.py:27-61`. Returns a list
    of arrays of shape (2, N_wl_lab) — [wavelength_um, albedo] per
    component.
    """
    input_file_path = os.environ.get("POSEIDON_input_data")  # noqa: SIM112
    if input_file_path is None:
        raise Exception(
            "POSEIDON cannot locate the input folder.\n"
            "Please set the 'POSEIDON_input_data' variable in your .bashrc "
            "or .bash_profile to point to the POSEIDON input folder."
        )

    surface_components = np.array(surface_components)
    out = []
    for component in surface_components:
        file_path = os.path.join(
            input_file_path, "surface_reflectivities", f"{component}.txt"
        )
        try:
            data = np.loadtxt(file_path).T
        except Exception as exc:
            raise Exception(
                f"Cannot load: {file_path}\nMake sure the txt file matches the "
                "surface components and is in the right folder!"
            ) from exc
        out.append(data)
    return out


def interpolate_surface_components(wl, surface_components, surface_component_albedos):
    """Linear interpolation of albedo onto model wavelength grid.

    Bit-equivalent port of POSEIDON `surfaces.py:63-99`. Raises if the
    model grid extends beyond any lab-data file's wavelength range.
    """
    surf_reflect_array = []
    for n in range(len(surface_component_albedos)):
        wavelength_txt_file = surface_component_albedos[n][0]
        albedo_txt_file = surface_component_albedos[n][1]

        if (np.min(wl) < np.min(wavelength_txt_file)) or (
            np.max(wl) > np.max(wavelength_txt_file)
        ):
            raise Exception(
                f"The wl grid exceeds the wavelengths of the albedo file: "
                f"{surface_components[n]} ({np.min(wavelength_txt_file)}, "
                f"{np.max(wavelength_txt_file)})"
            )
        f = interp1d(wavelength_txt_file, albedo_txt_file)
        surf_reflect_array.append(f(wl))
    return surf_reflect_array
