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

    Bit-equivalent port of POSEIDON `high_res.py:179-256`. Returns the
    residual array after `niter` cycles of one-component fits.
    """
    Npix, Nframes = data_array.shape
    residuals = data_array.copy()
    w = 1.0 / uncertainties**2

    for _ in range(niter):
        c = np.ones(Nframes)
        a = np.zeros(Npix)
        # 2-step alternating optimisation
        for _ in range(15):
            numer_a = (residuals * c[np.newaxis, :] * w).sum(axis=1)
            denom_a = (c[np.newaxis, :] ** 2 * w).sum(axis=1)
            a = numer_a / denom_a

            numer_c = (residuals * a[:, np.newaxis] * w).sum(axis=0)
            denom_c = (a[:, np.newaxis] ** 2 * w).sum(axis=0)
            c = numer_c / denom_c

        residuals -= np.outer(a, c)
    return residuals
