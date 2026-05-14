"""Stellar grid loaders (setup-only).

Setup-only: pysynphot / PyMSG / numpy / file I/O permitted. Must not
be called from inside `jit`. Allow-listed by the v1 source-grep gate.

Ports POSEIDON `stellar.py:63-211`:
- `load_stellar_pysynphot` — pysynphot CK04 / Phoenix ICAT grid lookup
- `open_pymsg_grid` + `load_stellar_pymsg` — PyMSG HDF5 grid

Both grids are env-gated (`PYSYN_CDBS` and `MSG_DIR` respectively).
The port keeps the same signatures POSEIDON exposes; CI uses
synthetic-spectrum fixtures.
"""

import numpy as np


def load_stellar_pysynphot(wl_out, T_eff, Met, log_g, stellar_grid="cbk04"):
    """pysynphot ICAT lookup → photospheric specific intensity on `wl_out`.

    Bit-equivalent port of POSEIDON `stellar.py:63-107`. Requires
    `$PYSYN_CDBS` to point at a pysynphot grid root.
    """
    try:
        import pysynphot as psyn
    except ImportError as exc:
        raise Exception(
            "pysynphot is required for load_stellar_pysynphot. "
            "Install pysynphot and set $PYSYN_CDBS."
        ) from exc

    if stellar_grid == "cbk04":
        sp = psyn.Icat("ck04models", T_eff, Met, log_g)
    elif stellar_grid == "phoenix":
        sp = psyn.Icat("phoenix", T_eff, Met, log_g)
    else:
        raise Exception(f"Stellar grid {stellar_grid!r} not supported")

    sp.convert("um")
    sp.convert("flam")
    wl_grid = sp.wave
    F_lambda = sp.flux

    # Interpolate to model grid.
    I_phot = np.interp(wl_out, wl_grid, F_lambda) / np.pi
    return I_phot


def open_pymsg_grid(stellar_grid):
    """Open a PyMSG SpecGrid (bit-equivalent to POSEIDON `stellar.py:110-150`)."""
    try:
        import pymsg
    except ImportError as exc:
        raise Exception(
            "PyMSG is required for open_pymsg_grid. Install PyMSG and set $MSG_DIR."
        ) from exc
    import os

    msg_dir = os.environ.get("MSG_DIR")
    if msg_dir is None:
        raise Exception(
            "PyMSG cannot locate $MSG_DIR. Set the env var to point at the "
            "PyMSG grids directory."
        )
    grid_path = os.path.join(msg_dir, "grids", f"{stellar_grid}.h5")
    return pymsg.SpecGrid(grid_path)


def load_stellar_pymsg(wl_out, specgrid, T_eff, Met, log_g, stellar_grid):
    """PyMSG SpecGrid lookup → photospheric I on `wl_out`.

    Bit-equivalent port of POSEIDON `stellar.py:153-211`.
    """
    x = {"Teff": T_eff, "[Fe/H]": Met, "log(g)": log_g}
    wl_lam = wl_out * 1.0e4  # μm → Å
    F_lambda = np.array(specgrid.flux(x, wl_lam))
    # Convert PyMSG erg/s/cm²/Å → W/m²/sr/m and divide by π for intensity.
    F_SI = F_lambda * 1e-7 / 1e-4 / 1e-10  # erg → J, cm⁻² → m⁻², Å⁻¹ → m⁻¹
    return F_SI / np.pi
