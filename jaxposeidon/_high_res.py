"""High-resolution-spectroscopy hot-path functions.

Ports POSEIDON `high_res.py:14-29, 179-256, 257-285, 319-336, 339-404,
440-450, 834-859, 862-882`.

0.5.16a shipped airtovac / vactoair / sysrem.

0.5.16b1 adds the data-prep + Doppler + CCF surface:
`fast_filter`, `fit_out_transit_spec`, `get_RV_range`, `find_nearest_idx`,
`cross_correlate`, `get_rot_kernel`, `remove_outliers`.

The PCA-based and h5py-based pieces (`make_data_cube`, `PCA_rebuild`,
`prepare_high_res_data`, `loglikelihood_PCA`, `loglikelihood_sysrem`,
`loglikelihood_high_res`) are deferred to 0.5.16b2.
"""

import numpy as np

# Speed of light in m/s (CODATA, SI-exact). Matches POSEIDON's
# `scipy.constants.c` used at `high_res.py:377, 564, 570, 686, 687, 927`.
_C_M_PER_S = 299792458.0


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


def fast_filter(flux, uncertainties, niter=15, Print=True):
    """Per-order SYSREM filtering across a (nord, nphi, npix) cube.

    Bit-equivalent port of POSEIDON `high_res.py:257-285`. Returns
    `(residuals, Us)` of shapes `(nord, nphi, npix)` and
    `(nord, nphi, niter + 1)`.
    """
    if Print:
        print(f"Filtering out systematics using SYSREM with {niter} iterations")
    nord, nphi, npix = flux.shape
    residuals = np.zeros((nord, nphi, npix))
    Us = np.zeros((nord, nphi, niter + 1))

    for i, order in enumerate(flux):
        stds = uncertainties[i]
        residual, U = sysrem(order, stds, niter)
        residuals[i] = residual
        Us[i] = U

    return residuals, Us


def fit_out_transit_spec(flux, transit_weight, degree=2, spec="median", Print=True):
    """Out-of-transit reference spectrum per order.

    Bit-equivalent port of POSEIDON `high_res.py:319-336`.
    """
    nord, nphi, npix = flux.shape
    spec_fit = np.zeros_like(flux)
    out_transit = transit_weight == 1

    for i in range(nord):
        if spec == "mean":
            mean_spec = np.mean(flux[i][out_transit], axis=0)
        elif spec == "median":
            mean_spec = np.median(flux[i][out_transit], axis=0)
        else:
            raise Exception('Error: Please select "mean", "median"')

        for j in range(nphi):
            spec_fit[i, j, :] = mean_spec

    return spec_fit


def get_RV_range(Kp_range, Vsys_range, phi):
    """Aggregate RV grid covering Kp × Vsys × phase.

    Bit-equivalent port of POSEIDON `high_res.py:339-344`.
    """
    RV_min = min(
        [
            np.min(Kp_range * np.sin(2 * np.pi * phi[i])) + np.min(Vsys_range)
            for i in range(len(phi))
        ]
    )
    RV_max = max(
        [
            np.max(Kp_range * np.sin(2 * np.pi * phi[i])) + np.max(Vsys_range)
            for i in range(len(phi))
        ]
    )
    RV_range = np.arange(RV_min, RV_max + 1)

    return RV_range


def find_nearest_idx(array, value):
    """Index of array's element closest to value.

    Bit-equivalent port of POSEIDON `high_res.py:440-450`.
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def cross_correlate(
    Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=True
):
    """Phase × RV → Kp × Vsys cross-correlation map.

    Bit-equivalent port of POSEIDON `high_res.py:347-404`. `data` is the
    per-dataset dict that POSEIDON writes via prepare_high_res_data: keys
    `uncertainties`, `residuals`, `phi`, `wl_grid`, and optionally
    `V_bary` and `transit_weight`.
    """
    if Print:
        import time as _time

        time0 = _time.time()
    uncertainties = data["uncertainties"]
    residuals = data["residuals"]
    phi = data["phi"]
    wl_grid = data["wl_grid"]

    try:
        V_bary = data["V_bary"]
    except KeyError:
        V_bary = np.zeros_like(phi)
    if "transit_weight" in data.keys():
        spectrum_type = "transmission"
        transit_weight = data["transit_weight"]
    else:
        spectrum_type = "emission"

    nord, nphi, npix = residuals.shape
    CCF_Kp_Vsys = np.zeros((len(Kp_range), len(Vsys_range)))

    nRV = len(RV_range)
    CCF_phase_RV = np.zeros((nphi, nRV))
    models_shifted = np.zeros((nRV, nord, npix))
    for RV_i, RV in enumerate(RV_range):
        for ord_i in range(nord):
            wl_slice = wl_grid[ord_i]
            delta_lambda = RV * 1e3 / _C_M_PER_S
            wl_shifted = wl * (1.0 + delta_lambda)
            F_p = np.interp(wl_slice, wl_shifted, planet_spectrum)
            models_shifted[RV_i, ord_i] = F_p

    if spectrum_type == "emission":
        m = models_shifted
    elif spectrum_type == "transmission":
        m = -models_shifted

    for phi_i in range(nphi):
        for RV_i in range(nRV):
            f = residuals[:, phi_i, :]
            CCF = np.sum(f[:, :] * m[RV_i, :, :] / uncertainties[:, phi_i, :] ** 2)
            CCF_phase_RV[phi_i, RV_i] += CCF

    if spectrum_type == "transmission":
        CCF_phase_RV = (1 - transit_weight[:, None]) * CCF_phase_RV

    for Kp_i, Kp in enumerate(Kp_range):
        for phi_i in range(nphi):
            RV = Kp * np.sin(2 * np.pi * phi[phi_i]) + Vsys_range + V_bary[phi_i]
            CCF_Kp_Vsys[Kp_i] += np.interp(RV, RV_range, CCF_phase_RV[phi_i])
    if Print:
        import time as _time

        time1 = _time.time()
        print(f"Cross correlation took {time1 - time0} seconds")
    return CCF_Kp_Vsys, CCF_phase_RV


def get_rot_kernel(V_sin_i, wl, W_conv):
    """Stellar rotational broadening kernel.

    Bit-equivalent port of POSEIDON `high_res.py:834-859`.
    """
    dRV = np.mean(2.0 * (wl[1:] - wl[0:-1]) / (wl[1:] + wl[0:-1])) * 2.998e5
    n_ker = int(W_conv)
    half_n_ker = (n_ker - 1) // 2
    rot_ker = np.zeros(n_ker)
    for ii in range(n_ker):
        ik = ii - half_n_ker
        x = ik * dRV / V_sin_i
        if np.abs(x) < 1.0:
            y = np.sqrt(1 - x**2)
            rot_ker[ii] = y
    rot_ker /= rot_ker.sum()

    return rot_ker


def remove_outliers(wl_grid, flux):
    """5σ outlier replacement via polyfit residuals.

    Bit-equivalent port of POSEIDON `high_res.py:862-882`.
    """
    nord, nphi, npix = flux.shape
    cleaned_flux = flux.copy()
    noutliers = 0
    for i in range(nord):
        for j in range(nphi):
            coeffs = np.polynomial.polynomial.polyfit(wl_grid[i], flux[i, j], 10)
            fitted_spectra = np.polynomial.polynomial.polyval(wl_grid[i], coeffs)
            std = np.std(flux[i, j] - fitted_spectra)
            outliers = np.abs(flux[i, j] - fitted_spectra) > 5 * std

            cleaned_flux[i, j, outliers] = np.interp(
                wl_grid[i, outliers], wl_grid[i, ~outliers], flux[i, j, ~outliers]
            )
            noutliers += np.sum(outliers)
    print(f"{noutliers} outliers removed from a total of {flux.size} pixels")

    return cleaned_flux
