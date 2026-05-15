"""High-resolution-spectroscopy hot-path functions.

Ports POSEIDON `high_res.py:14-29, 107-176, 179-256, 257-285, 288-316,
319-336, 339-404, 440-450, 503-609, 612-750, 753-831, 834-859, 862-882,
902-958`.

0.5.16a shipped airtovac / vactoair / sysrem.

0.5.16b1 added the data-prep + Doppler + CCF surface:
`fast_filter`, `fit_out_transit_spec`, `get_RV_range`, `find_nearest_idx`,
`cross_correlate`, `get_rot_kernel`, `remove_outliers`.

0.5.16b2 adds the multi-likelihood pipeline: `make_data_cube`,
`PCA_rebuild`, `prepare_high_res_data`, `loglikelihood_PCA`,
`loglikelihood_sysrem`, `loglikelihood_high_res`, `make_injection_data`.
"""

import os

import h5py
import numpy as np
from scipy import constants
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD

from jaxposeidon._constants import C_M_PER_S


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
            delta_lambda = RV * 1e3 / C_M_PER_S
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


def make_data_cube(data, n_components=4):
    """SVD/PCA background-subtract + 3-sigma clip per order.

    Bit-equivalent port of POSEIDON `high_res.py:288-306`. Returns
    `(data_scale, data_arr)` where `data_arr` is the residual cube with
    >3sigma deviations zeroed in-place.
    """
    nord, nphi, npix = data.shape

    data_scale = PCA_rebuild(data, n_components=n_components)
    data_arr = data - data_scale

    for i in range(nord):
        A = data_arr[i]
        sigma = np.std(A)
        median = np.median(A)
        loc = np.where(A > 3 * sigma + median)
        A[loc] = 0
        loc = np.where(A < -3 * sigma + median)
        A[loc] = 0
        data_arr[i] = A

    return data_scale, data_arr


def PCA_rebuild(flux, n_components=5):
    """Per-order TruncatedSVD reconstruction.

    Bit-equivalent port of POSEIDON `high_res.py:309-316`.
    """
    nord, nphi, npix = flux.shape
    rebuilt = np.zeros_like(flux)
    for i in range(nord):
        order = flux[i]
        svd = TruncatedSVD(n_components=n_components).fit(order)
        rebuilt[i] = svd.transform(order) @ svd.components_
    return rebuilt


def fit_uncertainties(flux, n_components=5, initial_guess=[0.1, 200], Print=True):
    """Poisson-style uncertainty fit per order via PCA residuals + Nelder-Mead.

    Bit-equivalent port of POSEIDON `high_res.py:53-75`.
    """
    if Print:
        print(f"Fitting Poisson uncertainties with {n_components} components")
    uncertainties = np.zeros(flux.shape)
    nord = len(flux)
    rebuilt = PCA_rebuild(flux, n_components=n_components)
    residuals = flux - rebuilt

    for i in range(nord):

        def neg_likelihood(x):
            a, b = x
            sigma = np.sqrt(a * flux[i] + b)
            loglikelihood = -0.5 * np.sum((residuals[i] / sigma) ** 2) - np.sum(
                np.log(sigma)
            )
            return -loglikelihood

        a, b = minimize(neg_likelihood, initial_guess, method="Nelder-Mead").x
        best_fit = np.sqrt(a * flux[i] + b)
        uncertainties[i] = best_fit

    return PCA_rebuild(uncertainties, n_components=n_components)


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

    Bit-equivalent port of POSEIDON `high_res.py:107-176`.
    """
    if spectrum_type == "transmission":
        if transit_weight is None:
            raise Exception(
                "Please provide transit_weight for transmission spectroscopy."
            )

    processed_data_path = os.path.join(data_dir, name, "data_processed.hdf5")

    # v1-grep-skip: setup-only output writer; v1.0.x moves to _lbl_table_loader-style writer
    _f_ctx = h5py.File(processed_data_path, "w")  # v1-grep-skip
    with _f_ctx as f:
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

    return


def loglikelihood_PCA(V_sys, K_p, d_phi, a, wl, planet_spectrum, star_spectrum, data):
    """PCA-based log-likelihood (Line 2019).

    Bit-equivalent port of POSEIDON `high_res.py:503-609`.
    """
    residuals = data["residuals"]
    flux = data["flux"]
    data_scale = flux - residuals
    phi = data["phi"]
    wl_grid = data["wl_grid"]

    try:
        V_bary = data["V_bary"]
    except KeyError:
        V_bary = np.zeros_like(phi)

    nord, nphi, npix = residuals.shape

    radial_velocity_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (phi + d_phi))
    delta_lambda_p = radial_velocity_p * 1e3 / constants.c

    K_s = 0.3229
    radial_velocity_s = V_sys + V_bary - K_s * np.sin(2 * np.pi * phi) * 0
    delta_lambda_s = radial_velocity_s * 1e3 / constants.c

    loglikelihood_sum = 0
    CCF_sum = 0
    for j in range(nord):
        wl_slice = wl_grid[j]
        F_p_F_s = np.zeros((nphi, npix))
        for i in range(nphi):
            wl_shifted_p = wl_slice * (1.0 - delta_lambda_p[i])
            F_p = np.interp(wl_shifted_p, wl, planet_spectrum)
            wl_shifted_s = wl_slice * (1.0 - delta_lambda_s[i])
            F_s = np.interp(wl_shifted_s, wl, star_spectrum)
            F_p_F_s[i, :] = F_p / F_s

        model_injected = (1 + F_p_F_s) * data_scale[j, :]

        svd = TruncatedSVD(n_components=4, n_iter=4, random_state=42).fit(
            model_injected
        )
        models_filtered = model_injected - (
            svd.transform(model_injected) @ svd.components_
        )

        for i in range(nphi):
            model_filtered = models_filtered[i] * a
            model_filtered -= model_filtered.mean()
            m2 = model_filtered.dot(model_filtered)
            planet_signal = residuals[j, i]
            f2 = planet_signal.dot(planet_signal)
            R = model_filtered.dot(planet_signal)
            CCF = R / np.sqrt(m2 * f2)
            CCF_sum += CCF
            loglikelihood_sum += -0.5 * npix * np.log((m2 + f2 - 2.0 * R) / npix)

    return loglikelihood_sum, CCF_sum


def loglikelihood_sysrem(
    V_sys, K_p, d_phi, a, b, wl, planet_spectrum, data, star_spectrum=None
):
    """SysRem-based log-likelihood (Gibson 2021/2022).

    Bit-equivalent port of POSEIDON `high_res.py:612-750`.
    """
    wl_grid = data["wl_grid"]
    residuals = data["residuals"]
    Bs = data["Bs"]
    phi = data["phi"]
    if star_spectrum is None:
        transit_weight = data["transit_weight"]
        max_transit_depth = np.max(1 - transit_weight)
    else:
        flux = data["flux"]
        flux_star = flux - residuals

    try:
        V_bary = data["V_bary"]
    except KeyError:
        V_bary = np.zeros_like(phi)

    uncertainties = data.get("uncertainties")

    nord, nphi, npix = residuals.shape

    N = residuals.size

    radial_velocity_p = V_sys + V_bary + K_p * np.sin(2 * np.pi * (phi + d_phi))
    radial_velocity_s = V_sys + V_bary + 0

    delta_lambda_p = radial_velocity_p * 1e3 / constants.c
    delta_lambda_s = radial_velocity_s * 1e3 / constants.c

    loglikelihood_sum = 0
    if b is not None:
        loglikelihood_sum -= N * np.log(b)

    for i in range(nord):
        wl_slice = wl_grid[i]

        models_shifted = np.zeros((nphi, npix))

        for j in range(nphi):
            wl_shifted_p = wl_slice * (1.0 - delta_lambda_p[j])
            F_p = np.interp(wl_shifted_p, wl, planet_spectrum * a)
            if star_spectrum is None:
                models_shifted[j] = (1 - transit_weight[j]) / max_transit_depth * (
                    -F_p
                ) + 1
                models_shifted[j] /= np.median(models_shifted[j])
            else:
                wl_shifted_s = wl_slice * (1.0 - delta_lambda_s[j])
                F_s = np.interp(wl_shifted_s, wl, star_spectrum)
                models_shifted[j] = F_p / F_s * flux_star[i, j]

        B = Bs[i]
        models_filtered = models_shifted - B @ models_shifted

        if b is not None:
            for j in range(nphi):
                m = models_filtered[j] / uncertainties[i, j]
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -0.5 * (m2 + f2 - 2.0 * CCF) / (b**2)
                loglikelihood_sum += loglikelihood

        elif uncertainties is not None:
            for j in range(nphi):
                m = models_filtered[j] / uncertainties[i, j]
                m2 = m.dot(m)
                f = residuals[i, j] / uncertainties[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -npix / 2 * np.log((m2 + f2 - 2.0 * CCF) / npix)
                loglikelihood_sum += loglikelihood

        else:
            for j in range(nphi):
                m = models_filtered[j]
                m2 = m.dot(m)
                f = residuals[i, j]
                f2 = f.dot(f)
                CCF = f.dot(m)
                loglikelihood = -npix / 2 * np.log((m2 + f2 - 2.0 * CCF) / npix)
                loglikelihood_sum += loglikelihood

    return loglikelihood_sum


def loglikelihood_high_res(
    wl,
    planet_spectrum,
    star_spectrum,
    data,
    spectrum_type,
    method,
    high_res_params,
    high_res_param_names,
):
    """Multi-dataset / multi-method log-likelihood dispatch.

    Bit-equivalent port of POSEIDON `high_res.py:753-831`.
    """
    K_p = high_res_params[np.where(high_res_param_names == "K_p")[0][0]]
    V_sys = high_res_params[np.where(high_res_param_names == "V_sys")[0][0]]

    if "log_alpha_HR" in high_res_param_names:
        a = (
            10
            ** high_res_params[np.where(high_res_param_names == "log_alpha_HR")[0][0]]
        )
    elif "alpha_HR" in high_res_param_names:
        a = high_res_params[np.where(high_res_param_names == "alpha_HR")[0][0]]
    else:
        a = 1

    if "Delta_phi" in high_res_param_names:
        d_phi = high_res_params[np.where(high_res_param_names == "Delta_phi")[0][0]]
    else:
        d_phi = 0

    if "W_conv" in high_res_param_names:
        W_conv = high_res_params[np.where(high_res_param_names == "W_conv")[0][0]]
    else:
        W_conv = None

    if "beta_HR" in high_res_param_names:
        b = high_res_params[np.where(high_res_param_names == "beta_HR")[0][0]]
    else:
        b = None

    if spectrum_type == "emission":
        if W_conv is not None:
            F_p = gaussian_filter1d(planet_spectrum, W_conv)
            F_s = gaussian_filter1d(star_spectrum, W_conv)
        else:
            F_p = planet_spectrum
            F_s = star_spectrum
        loglikelihood = 0
        for key in data.keys():
            if method == "sysrem":
                loglikelihood += loglikelihood_sysrem(
                    V_sys, K_p, d_phi, a, b, wl, F_p, data[key], F_s
                )
            elif method == "PCA":
                loglikelihood, _ = loglikelihood_PCA(
                    V_sys, K_p, d_phi, a, wl, F_p, F_s, data[key]
                )
            else:
                raise Exception(
                    "Emission spectroscopy only supports sysrem and PCA for now."
                )
        return loglikelihood

    elif spectrum_type == "transmission":
        if method != "sysrem":
            raise Exception(
                "Transmission spectroscopy only supports fast filtering with "
                "sysrem (Gibson et al. 2022)."
            )
        if W_conv is not None:
            F_p = gaussian_filter1d(planet_spectrum, W_conv)
        else:
            F_p = planet_spectrum
        loglikelihood = 0
        for key in data.keys():
            loglikelihood += loglikelihood_sysrem(
                V_sys, K_p, d_phi, a, b, wl, F_p, data[key]
            )
        return loglikelihood
    else:
        raise Exception("Spectrum type should be 'emission' or 'transmission'.")


def make_injection_data(
    data,
    data_dir,
    name,
    wl,
    planet_spectrum,
    K_p,
    V_sys,
    method,
    a=None,
    continuum=None,
    W_conv=None,
    star_spectrum=None,
):
    """Inject a forward-model spectrum into raw flux and re-prepare.

    Bit-equivalent port of POSEIDON `high_res.py:902-958`.
    """
    residuals = data["residuals"]
    flux = data["flux"]
    wl_grid = data["wl_grid"]
    phi = data["phi"]

    nord, nphi, npix = residuals.shape
    if continuum is not None and a is not None:
        planet_spectrum = (planet_spectrum - continuum) * a + continuum
    if W_conv is not None:
        planet_spectrum = gaussian_filter1d(planet_spectrum, W_conv)
    emission = star_spectrum is not None

    if emission:
        spectrum_type = "emission"
        if W_conv is not None:
            star_spectrum = gaussian_filter1d(star_spectrum, W_conv)
        transit_weight = None
    else:
        spectrum_type = "transmission"
        transit_weight = data["transit_weight"]
        max_transit_depth = np.max(1 - transit_weight)
    radial_velocity = V_sys + K_p * np.sin(2 * np.pi * phi)
    delta_lambda = radial_velocity * 1e3 / constants.c

    F_p_F_s = np.zeros((nord, nphi, npix))
    F_p = np.zeros((nord, nphi, npix))

    for i in range(nord):
        wl_slice = wl_grid[i].copy()
        for j in range(nphi):
            wl_shifted_p = wl_slice * (1.0 - delta_lambda[j])
            if emission:
                F_p_F_s[i, j, :] = np.interp(
                    wl_shifted_p, wl, planet_spectrum
                ) / np.interp(wl_slice, wl, star_spectrum)
            else:
                F_p[i, j, :] = (
                    -np.interp(wl_shifted_p, wl, planet_spectrum)
                    * (1 - transit_weight[j])
                    / max_transit_depth
                    + 1
                )

    if emission:
        data_injected = (1 + F_p_F_s) * (flux - residuals)
    else:
        data_injected = F_p * flux

    if method.lower() == "pca":
        uncertainties = None
    elif method.lower() == "sysrem":
        uncertainties = fit_uncertainties(
            data_injected, initial_guess=[0.1, np.mean(data_injected)]
        )

    prepare_high_res_data(
        data_dir,
        name,
        spectrum_type,
        method,
        data_injected,
        wl_grid,
        phi,
        uncertainties,
        transit_weight,
    )

    return


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
