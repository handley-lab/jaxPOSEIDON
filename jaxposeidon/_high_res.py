"""High-resolution-spectroscopy hot-path functions.

Ports POSEIDON `high_res.py:14-29` and the iterative-detrending stages
that consume per-order data cubes.

This first slice is the small numeric utilities (air↔vacuum conversion,
sysrem iterative detrending). The larger orchestration around
prepare_high_res_data + per-order CCF + multiple likelihood
prescriptions lives in POSEIDON `high_res.py:107+` and depends on a
specific data layout that's only meaningful when real high-res
spectra are available (env-gated).
"""

import numpy as np


def airtovac(wlum):
    """Air → vacuum wavelength conversion (μm).

    Bit-equivalent port of POSEIDON `high_res.py:14-20`.
    """
    wl = wlum * 1e4
    s = 1e4 / wl
    n = 1 + (
        0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    return wl * n * 1e-4


def vactoair(wlum):
    """Vacuum → air wavelength conversion (μm).

    Bit-equivalent port of POSEIDON `high_res.py:23-28`.
    """
    wl = wlum * 1e4
    s = 1e4 / wl
    n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998 / (38.9 - s**2)
    return wl / n * 1e-4


def sysrem(data_array, uncertainties, niter=15):
    """Iterative `sysrem` detrending (Tamuz, Mazeh & Zucker 2005).

    Bit-equivalent port of POSEIDON `high_res.py:179-256`. Returns
    `(residuals, U)`: filtered (nphi × npix) residuals and basis vectors
    `U` of shape `(nphi, niter+1)`.
    """
    data_array = data_array.T
    uncertainties = uncertainties.T
    npix, nphi = data_array.shape

    residuals = np.zeros((npix, nphi))
    for i, light_curve in enumerate(data_array):
        residuals[i] = light_curve - np.median(light_curve)

    U = np.zeros((nphi, niter + 1))

    for i in range(niter):
        w = np.zeros(npix)
        u = np.ones(nphi)

        for _ in range(10):
            for pix in range(npix):
                err_squared = uncertainties[pix] ** 2
                numerator = np.sum(u * residuals[pix] / err_squared)
                denominator = np.sum(u**2 / err_squared)
                w[pix] = numerator / denominator

            for phi in range(nphi):
                err_squared = uncertainties[:, phi] ** 2
                numerator = np.sum(w * residuals[:, phi] / err_squared)
                denominator = np.sum(w**2 / err_squared)
                u[phi] = numerator / denominator

        systematic = np.zeros((npix, nphi))
        for pix in range(npix):
            for phi in range(nphi):
                systematic[pix, phi] = u[phi] * w[pix]

        residuals = residuals - systematic
        U[:, i] = u

    U[:, -1] = np.ones(nphi)

    return residuals.T, U
