"""Setup-only writer for processed high-resolution data files.

Allow-listed by the source-grep gate (uses h5py + numpy + scipy +
sklearn for one-off per-dataset processing — must NOT be called from
inside ``jit``). The hot-path likelihood kernels in
``_jax_high_res.py`` consume the prepared tensors directly without
touching the file format.
"""

import os

import h5py
import numpy as np

from jaxposeidon._high_res import (
    fast_filter,
    fit_out_transit_spec,
    make_data_cube,
)


def prepare_high_res_data(
    data_dir,
    name,
    spectrum_type,
    method,
    flux,
    wl_grid,
    phi,
    uncertainties=None,
    transit_weight=None,
    V_bary=None,
    pca_ncomp=4,
    sysrem_niter=15,
):
    """Write `<data_dir>/<name>/data_processed.hdf5` per POSEIDON convention.

    Setup-only — uses h5py.File for output. Mirrors POSEIDON
    `high_res.py:107-176`.
    """
    if spectrum_type == "transmission":
        if transit_weight is None:
            raise Exception(
                "Please provide transit_weight for transmission spectroscopy."
            )

    processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")

    with h5py.File(processed_data_path, "w") as f:
        print(f"Creating processed data at {processed_data_path}")
        f.create_dataset("phi", data=phi)
        f.create_dataset("wl_grid", data=wl_grid)
        if V_bary is not None:
            f.create_dataset("V_bary", data=V_bary)

        nord, nphi, npix = flux.shape

        if spectrum_type == "emission":
            if method.lower() == "pca":
                _, residuals = make_data_cube(flux, pca_ncomp)
            elif method.lower() == "sysrem":
                residuals, Us = fast_filter(flux, uncertainties, sysrem_niter)
                Bs = np.zeros((nord, nphi, nphi))
                for i in range(nord):
                    U = Us[i]
                    L = np.diag(1 / np.mean(uncertainties[i], axis=-1))
                    B = U @ np.linalg.pinv(L @ U) @ L
                    Bs[i] = B
                f.create_dataset("Bs", data=Bs)
                f.create_dataset("uncertainties", data=uncertainties)
            f.create_dataset("flux", data=flux)
            f.create_dataset("residuals", data=residuals)

        elif spectrum_type == "transmission":
            if method.lower() == "sysrem":
                median = fit_out_transit_spec(flux, transit_weight, spec="median")
                flux /= median
                uncertainties /= median
                residuals, Us = fast_filter(flux, uncertainties, sysrem_niter)
                Bs = np.zeros((nord, nphi, nphi))
                for i in range(nord):
                    U = Us[i]
                    L = np.diag(1 / np.mean(uncertainties[i], axis=-1))
                    B = U @ np.linalg.pinv(L @ U) @ L
                    Bs[i] = B
                f.create_dataset("Bs", data=Bs)
                f.create_dataset("residuals", data=residuals)
                f.create_dataset("uncertainties", data=uncertainties)
                f.create_dataset("transit_weight", data=transit_weight)
