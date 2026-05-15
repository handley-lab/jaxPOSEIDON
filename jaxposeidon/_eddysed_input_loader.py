"""Eddysed / PICASO / VIRGA cloud-output file loader.

Setup-only module: numpy and h5py file I/O are permitted; never imported
inside any JAX-traced or `jit`-decorated function.

POSEIDON itself does not ship a dedicated reader for eddysed file
output (see `POSEIDON/core.py:1685-1700`, `POSEIDON/parameters.py:2440-2473`):
the user is expected to populate `cloud_params` (or equivalently the
atmosphere dict keys `kappa_cloud_eddysed`, `g_cloud_eddysed`,
`w_cloud_eddysed`) with arrays drawn from a separate PICASO/VIRGA
calculation. This loader provides the thin file-I/O wrapper that
external pipelines (PICASO, VIRGA) emit when writing their per-layer
per-wavelength cloud opacity / single-scattering albedo / asymmetry
parameter to an HDF5 (or .npz) container.

The HDF5 schema mirrors PICASO's per-`atmosphere.compute` output:

    /kappa_cloud  (N_layers, N_wl)  -- cloud opacity [cm^-1]
    /g_cloud      (N_layers, N_wl)  -- asymmetry parameter (dimensionless)
    /w_cloud      (N_layers, N_wl)  -- single-scattering albedo (dimensionless)

Optional `/P` and `/wavelength` datasets are read but not validated
against the model's pressure / wavelength grids: the caller is
responsible for grid alignment (matching POSEIDON's own no-validation
convention).
"""

import numpy as np


def read_eddysed_file(path):
    """Read an eddysed/PICASO/VIRGA cloud-output HDF5 file.

    Args:
        path: filesystem path to the HDF5 container.

    Returns:
        dict with keys 'kappa_cloud', 'g_cloud', 'w_cloud' (each
        np.ndarray, shape (N_layers, N_wl)) plus optional 'P' and
        'wavelength' if present in the file.

    The returned arrays are passed verbatim into POSEIDON's atmosphere
    dict (`atmosphere['kappa_cloud_eddysed']` etc.); see
    `POSEIDON/core.py:1685-1700` for the runtime reshape that consumes
    them.
    """
    import h5py

    out = {}
    with h5py.File(path, "r") as f:
        out["kappa_cloud"] = np.asarray(f["kappa_cloud"][...])
        out["g_cloud"] = np.asarray(f["g_cloud"][...])
        out["w_cloud"] = np.asarray(f["w_cloud"][...])
        if "P" in f:
            out["P"] = np.asarray(f["P"][...])
        if "wavelength" in f:
            out["wavelength"] = np.asarray(f["wavelength"][...])
    return out


def reshape_eddysed_for_atmosphere(
    kappa_cloud, g_cloud, w_cloud, N_sectors=1, N_zones=1
):
    """Broadcast 2-D (N_layers, N_wl) arrays to POSEIDON's 4-D layout.

    POSEIDON's `kappa_cloud` array inside `compute_spectrum` is
    (N_layers, N_sectors, N_zones, N_wl). This helper inserts the
    sector / zone axes so a PICASO 1-D output (which has no terminator
    structure) can drop into the eddysed dispatch branch
    (`core.py:1685-1700`).
    """
    kappa = np.asarray(kappa_cloud)
    g = np.asarray(g_cloud)
    w = np.asarray(w_cloud)
    if kappa.ndim == 2:
        N_layers, N_wl = kappa.shape
        kappa = np.broadcast_to(
            kappa[:, None, None, :], (N_layers, N_sectors, N_zones, N_wl)
        ).copy()
        g = np.broadcast_to(
            g[:, None, None, :], (N_layers, N_sectors, N_zones, N_wl)
        ).copy()
        w = np.broadcast_to(
            w[:, None, None, :], (N_layers, N_sectors, N_zones, N_wl)
        ).copy()
    return kappa, g, w
