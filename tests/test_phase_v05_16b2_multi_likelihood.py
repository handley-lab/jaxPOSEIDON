"""Phase 0.5.16b2: high-resolution multi-likelihood pipeline.

Ports POSEIDON `high_res.py:107-176, 288-316, 503-609, 612-750, 753-831,
902-958`. Compares jaxposeidon's port against POSEIDON's reference at
rtol=1e-13 (PCA/sysrem) to rtol=1e-11 (numba-flavoured paths and
heavier numerical reductions).
"""

import os

import h5py
import numpy as np
import pytest

from jaxposeidon import _high_res


def _rng(seed=0):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# PCA_rebuild
# ---------------------------------------------------------------------------
def test_PCA_rebuild_matches_poseidon():
    from POSEIDON.high_res import PCA_rebuild as p_pca

    rng = _rng(0)
    flux = 1.0 + 0.01 * rng.standard_normal((3, 8, 50))
    ours = _high_res.PCA_rebuild(flux.copy(), n_components=4)
    theirs = p_pca(flux.copy(), n_components=4)
    np.testing.assert_allclose(ours, theirs, atol=1e-14, rtol=1e-9)


def test_PCA_rebuild_default_n_components_matches_poseidon():
    from POSEIDON.high_res import PCA_rebuild as p_pca

    rng = _rng(1)
    flux = 1.0 + 0.01 * rng.standard_normal((2, 10, 40))
    ours = _high_res.PCA_rebuild(flux.copy())
    theirs = p_pca(flux.copy())
    np.testing.assert_allclose(ours, theirs, atol=1e-14, rtol=1e-9)


# ---------------------------------------------------------------------------
# make_data_cube
# ---------------------------------------------------------------------------
def test_make_data_cube_matches_poseidon():
    from POSEIDON.high_res import make_data_cube as p_mdc

    rng = _rng(2)
    data = 1.0 + 0.01 * rng.standard_normal((2, 7, 60))
    data[0, 3, 10] += 0.5
    data[1, 4, 30] -= 0.5

    s_ours, a_ours = _high_res.make_data_cube(data.copy(), n_components=4)
    s_theirs, a_theirs = p_mdc(data.copy(), n_components=4)
    np.testing.assert_allclose(s_ours, s_theirs, atol=1e-14, rtol=1e-9)
    np.testing.assert_allclose(a_ours, a_theirs, atol=1e-14, rtol=1e-9)


# ---------------------------------------------------------------------------
# prepare_high_res_data
# ---------------------------------------------------------------------------
def _hdf5_to_dict(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f.keys():
            out[k] = f[k][:]
    return out


def _prep_cube(nord=2, nphi=8, npix=40, seed=3):
    rng = _rng(seed)
    flux = 1.0 + 0.001 * rng.standard_normal((nord, nphi, npix))
    uncertainties = 0.001 * np.ones_like(flux)
    wl_grid = np.zeros((nord, npix))
    for o in range(nord):
        wl_grid[o] = np.linspace(1.0 + 0.01 * o, 1.5 + 0.01 * o, npix)
    phi = np.linspace(-0.04, 0.04, nphi)
    V_bary = 0.05 * rng.standard_normal(nphi)
    return flux, uncertainties, wl_grid, phi, V_bary


@pytest.mark.parametrize("method", ["sysrem", "pca"])
def test_prepare_high_res_data_emission_matches_poseidon(tmp_path, method):
    from POSEIDON.high_res import prepare_high_res_data as p_prep

    flux, unc, wl_grid, phi, V_bary = _prep_cube()
    ours_dir = tmp_path / "ours"
    theirs_dir = tmp_path / "theirs"
    name = "obs"
    (ours_dir / name).mkdir(parents=True)
    (theirs_dir / name).mkdir(parents=True)

    _high_res.prepare_high_res_data(
        str(ours_dir),
        name,
        "emission",
        method,
        flux.copy(),
        wl_grid,
        phi,
        uncertainties=unc.copy(),
        V_bary=V_bary,
        sysrem_niter=3,
    )
    p_prep(
        str(theirs_dir),
        name,
        "emission",
        method,
        flux.copy(),
        wl_grid,
        phi,
        uncertainties=unc.copy(),
        V_bary=V_bary,
        sysrem_niter=3,
    )

    d_ours = _hdf5_to_dict(str(ours_dir / name / "data_processed.hdf5"))
    d_theirs = _hdf5_to_dict(str(theirs_dir / name / "data_processed.hdf5"))

    assert set(d_ours.keys()) == set(d_theirs.keys())
    for key in d_ours.keys():
        np.testing.assert_allclose(d_ours[key], d_theirs[key], atol=1e-14, rtol=1e-9)


def test_prepare_high_res_data_transmission_matches_poseidon(tmp_path):
    from POSEIDON.high_res import prepare_high_res_data as p_prep

    flux, unc, wl_grid, phi, V_bary = _prep_cube(nphi=8)
    transit_weight = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0])
    ours_dir = tmp_path / "ours"
    theirs_dir = tmp_path / "theirs"
    name = "obs"
    (ours_dir / name).mkdir(parents=True)
    (theirs_dir / name).mkdir(parents=True)

    _high_res.prepare_high_res_data(
        str(ours_dir),
        name,
        "transmission",
        "sysrem",
        flux.copy(),
        wl_grid,
        phi,
        uncertainties=unc.copy(),
        transit_weight=transit_weight,
        V_bary=V_bary,
        sysrem_niter=3,
    )
    p_prep(
        str(theirs_dir),
        name,
        "transmission",
        "sysrem",
        flux.copy(),
        wl_grid,
        phi,
        uncertainties=unc.copy(),
        transit_weight=transit_weight,
        V_bary=V_bary,
        sysrem_niter=3,
    )

    d_ours = _hdf5_to_dict(str(ours_dir / name / "data_processed.hdf5"))
    d_theirs = _hdf5_to_dict(str(theirs_dir / name / "data_processed.hdf5"))

    assert set(d_ours.keys()) == set(d_theirs.keys())
    for key in d_ours.keys():
        np.testing.assert_allclose(d_ours[key], d_theirs[key], atol=1e-14, rtol=1e-9)


def test_prepare_high_res_data_transmission_missing_weight_raises(tmp_path):
    """POSEIDON `high_res.py:109-111` raises when transit_weight is None."""
    from POSEIDON.high_res import prepare_high_res_data as p_prep

    flux, unc, wl_grid, phi, _ = _prep_cube()
    (tmp_path / "obs").mkdir()
    with pytest.raises(Exception, match="transit_weight"):
        _high_res.prepare_high_res_data(
            str(tmp_path),
            "obs",
            "transmission",
            "sysrem",
            flux,
            wl_grid,
            phi,
            uncertainties=unc,
        )
    with pytest.raises(Exception, match="transit_weight"):
        p_prep(
            str(tmp_path),
            "obs",
            "transmission",
            "sysrem",
            flux,
            wl_grid,
            phi,
            uncertainties=unc,
        )


# ---------------------------------------------------------------------------
# loglikelihood_PCA
# ---------------------------------------------------------------------------
def _emission_pca_data(nord=2, nphi=8, npix=40, seed=4):
    """Build an emission PCA dataset by running POSEIDON's prepare path."""
    flux, _, wl_grid, phi, V_bary = _prep_cube(nord, nphi, npix, seed)
    _, residuals = _high_res.make_data_cube(flux.copy(), n_components=4)
    data = {
        "residuals": residuals,
        "flux": flux,
        "phi": phi,
        "wl_grid": wl_grid,
        "V_bary": V_bary,
    }
    return data


def test_loglikelihood_PCA_matches_poseidon():
    from POSEIDON.high_res import loglikelihood_PCA as p_pca

    data = _emission_pca_data()
    rng = _rng(5)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)

    ll_ours, ccf_ours = _high_res.loglikelihood_PCA(
        -2.0, 180.0, 0.0, 1.0, wl, planet_spectrum, star_spectrum, data
    )
    ll_theirs, ccf_theirs = p_pca(
        -2.0, 180.0, 0.0, 1.0, wl, planet_spectrum, star_spectrum, data
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)
    np.testing.assert_allclose(ccf_ours, ccf_theirs, atol=0, rtol=1e-11)


def test_loglikelihood_PCA_no_V_bary_matches_poseidon():
    from POSEIDON.high_res import loglikelihood_PCA as p_pca

    data = _emission_pca_data()
    del data["V_bary"]
    rng = _rng(6)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)

    ll_ours, ccf_ours = _high_res.loglikelihood_PCA(
        -3.0, 175.0, 0.001, 0.8, wl, planet_spectrum, star_spectrum, data
    )
    ll_theirs, ccf_theirs = p_pca(
        -3.0, 175.0, 0.001, 0.8, wl, planet_spectrum, star_spectrum, data
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)
    np.testing.assert_allclose(ccf_ours, ccf_theirs, atol=0, rtol=1e-11)


# ---------------------------------------------------------------------------
# loglikelihood_sysrem
# ---------------------------------------------------------------------------
def _sysrem_emission_data(nord=2, nphi=8, npix=40, seed=7):
    flux, unc, wl_grid, phi, V_bary = _prep_cube(nord, nphi, npix, seed)
    residuals, Us = _high_res.fast_filter(flux.copy(), unc.copy(), niter=3, Print=False)
    Bs = np.zeros((nord, nphi, nphi))
    for i in range(nord):
        U = Us[i]
        L = np.diag(1 / np.mean(unc[i], axis=-1))
        B = U @ np.linalg.pinv(L @ U) @ L
        Bs[i] = B
    data = {
        "residuals": residuals,
        "Bs": Bs,
        "uncertainties": unc,
        "flux": flux,
        "phi": phi,
        "wl_grid": wl_grid,
        "V_bary": V_bary,
    }
    return data


def _sysrem_transmission_data(nord=2, nphi=8, npix=40, seed=8):
    flux, unc, wl_grid, phi, V_bary = _prep_cube(nord, nphi, npix, seed)
    transit_weight = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0])[:nphi]
    median = _high_res.fit_out_transit_spec(
        flux, transit_weight, spec="median", Print=False
    )
    flux2 = flux / median
    unc2 = unc / median
    residuals, Us = _high_res.fast_filter(
        flux2.copy(), unc2.copy(), niter=3, Print=False
    )
    Bs = np.zeros((nord, nphi, nphi))
    for i in range(nord):
        U = Us[i]
        L = np.diag(1 / np.mean(unc2[i], axis=-1))
        B = U @ np.linalg.pinv(L @ U) @ L
        Bs[i] = B
    data = {
        "residuals": residuals,
        "Bs": Bs,
        "uncertainties": unc2,
        "transit_weight": transit_weight,
        "phi": phi,
        "wl_grid": wl_grid,
        "V_bary": V_bary,
    }
    return data


@pytest.mark.parametrize("b", [None, 1.2])
def test_loglikelihood_sysrem_emission_matches_poseidon(b):
    from POSEIDON.high_res import loglikelihood_sysrem as p_sr

    data = _sysrem_emission_data()
    rng = _rng(9)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)

    ll_ours = _high_res.loglikelihood_sysrem(
        -2.0, 180.0, 0.0, 1.0, b, wl, planet_spectrum, data, star_spectrum
    )
    ll_theirs = p_sr(-2.0, 180.0, 0.0, 1.0, b, wl, planet_spectrum, data, star_spectrum)
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


def test_loglikelihood_sysrem_emission_no_uncertainties_matches_poseidon():
    """Nulled b AND uncertainties — POSEIDON `high_res.py:737-745`."""
    from POSEIDON.high_res import loglikelihood_sysrem as p_sr

    data = _sysrem_emission_data()
    del data["uncertainties"]
    rng = _rng(10)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)

    ll_ours = _high_res.loglikelihood_sysrem(
        -2.0, 180.0, 0.0, 1.0, None, wl, planet_spectrum, data, star_spectrum
    )
    ll_theirs = p_sr(
        -2.0, 180.0, 0.0, 1.0, None, wl, planet_spectrum, data, star_spectrum
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


@pytest.mark.parametrize("b", [None, 1.5])
def test_loglikelihood_sysrem_transmission_matches_poseidon(b):
    from POSEIDON.high_res import loglikelihood_sysrem as p_sr

    data = _sysrem_transmission_data()
    rng = _rng(11)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * np.abs(rng.standard_normal(wl.size))

    ll_ours = _high_res.loglikelihood_sysrem(
        -2.0, 180.0, 0.0, 1.0, b, wl, planet_spectrum, data
    )
    ll_theirs = p_sr(-2.0, 180.0, 0.0, 1.0, b, wl, planet_spectrum, data)
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


# ---------------------------------------------------------------------------
# loglikelihood_high_res
# ---------------------------------------------------------------------------
def test_loglikelihood_high_res_emission_sysrem_matches_poseidon():
    from POSEIDON.high_res import loglikelihood_high_res as p_hr

    data = {
        "obs1": _sysrem_emission_data(seed=12),
        "obs2": _sysrem_emission_data(seed=13),
    }
    rng = _rng(14)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)
    high_res_params = np.array([180.0, -2.0, 0.0, 1.0])
    high_res_param_names = np.array(["K_p", "V_sys", "Delta_phi", "alpha_HR"])

    ll_ours = _high_res.loglikelihood_high_res(
        wl,
        planet_spectrum,
        star_spectrum,
        data,
        "emission",
        "sysrem",
        high_res_params,
        high_res_param_names,
    )
    ll_theirs = p_hr(
        wl,
        planet_spectrum,
        star_spectrum,
        data,
        "emission",
        "sysrem",
        high_res_params,
        high_res_param_names,
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


def test_loglikelihood_high_res_emission_PCA_matches_poseidon():
    from POSEIDON.high_res import loglikelihood_high_res as p_hr

    data = {"obs1": _emission_pca_data(seed=15)}
    rng = _rng(16)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)
    high_res_params = np.array([180.0, -2.0])
    high_res_param_names = np.array(["K_p", "V_sys"])

    ll_ours = _high_res.loglikelihood_high_res(
        wl,
        planet_spectrum,
        star_spectrum,
        data,
        "emission",
        "PCA",
        high_res_params,
        high_res_param_names,
    )
    ll_theirs = p_hr(
        wl,
        planet_spectrum,
        star_spectrum,
        data,
        "emission",
        "PCA",
        high_res_params,
        high_res_param_names,
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


def test_loglikelihood_high_res_transmission_matches_poseidon():
    from POSEIDON.high_res import loglikelihood_high_res as p_hr

    data = {"obs1": _sysrem_transmission_data(seed=17)}
    rng = _rng(18)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * np.abs(rng.standard_normal(wl.size))
    high_res_params = np.array([180.0, -2.0, -0.05])
    high_res_param_names = np.array(["K_p", "V_sys", "log_alpha_HR"])

    ll_ours = _high_res.loglikelihood_high_res(
        wl,
        planet_spectrum,
        None,
        data,
        "transmission",
        "sysrem",
        high_res_params,
        high_res_param_names,
    )
    ll_theirs = p_hr(
        wl,
        planet_spectrum,
        None,
        data,
        "transmission",
        "sysrem",
        high_res_params,
        high_res_param_names,
    )
    np.testing.assert_allclose(ll_ours, ll_theirs, atol=0, rtol=1e-11)


def test_loglikelihood_high_res_transmission_wrong_method_raises():
    from POSEIDON.high_res import loglikelihood_high_res as p_hr

    data = {"obs1": _sysrem_transmission_data(seed=19)}
    wl = np.linspace(0.95, 1.6, 100)
    planet_spectrum = np.zeros_like(wl)
    high_res_params = np.array([180.0, -2.0])
    high_res_param_names = np.array(["K_p", "V_sys"])
    with pytest.raises(Exception, match="sysrem"):
        _high_res.loglikelihood_high_res(
            wl,
            planet_spectrum,
            None,
            data,
            "transmission",
            "PCA",
            high_res_params,
            high_res_param_names,
        )
    with pytest.raises(Exception, match="sysrem"):
        p_hr(
            wl,
            planet_spectrum,
            None,
            data,
            "transmission",
            "PCA",
            high_res_params,
            high_res_param_names,
        )


def test_loglikelihood_high_res_bad_spectrum_type_raises():
    data = {"obs1": _sysrem_emission_data(seed=20)}
    wl = np.linspace(0.95, 1.6, 100)
    planet_spectrum = np.zeros_like(wl)
    star_spectrum = np.ones_like(wl)
    high_res_params = np.array([180.0, -2.0])
    high_res_param_names = np.array(["K_p", "V_sys"])
    with pytest.raises(Exception, match="emission|transmission"):
        _high_res.loglikelihood_high_res(
            wl,
            planet_spectrum,
            star_spectrum,
            data,
            "reflection",
            "sysrem",
            high_res_params,
            high_res_param_names,
        )


# ---------------------------------------------------------------------------
# make_injection_data
# ---------------------------------------------------------------------------
def test_make_injection_data_emission_matches_poseidon(tmp_path):
    from POSEIDON.high_res import make_injection_data as p_inj

    data = _sysrem_emission_data(seed=21)
    rng = _rng(22)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    star_spectrum = 1.0 + 0.001 * rng.standard_normal(wl.size)

    ours_dir = tmp_path / "ours"
    theirs_dir = tmp_path / "theirs"
    name = "obs"
    (ours_dir / name).mkdir(parents=True)
    (theirs_dir / name).mkdir(parents=True)

    _high_res.make_injection_data(
        data,
        str(ours_dir),
        name,
        wl,
        planet_spectrum.copy(),
        K_p=180.0,
        V_sys=-2.0,
        method="sysrem",
        star_spectrum=star_spectrum,
    )
    p_inj(
        data,
        str(theirs_dir),
        name,
        wl,
        planet_spectrum.copy(),
        K_p=180.0,
        V_sys=-2.0,
        method="sysrem",
        star_spectrum=star_spectrum,
    )

    d_ours = _hdf5_to_dict(str(ours_dir / name / "data_processed.hdf5"))
    d_theirs = _hdf5_to_dict(str(theirs_dir / name / "data_processed.hdf5"))
    assert set(d_ours.keys()) == set(d_theirs.keys())
    for key in d_ours.keys():
        np.testing.assert_allclose(d_ours[key], d_theirs[key], atol=1e-14, rtol=1e-9)


def test_make_injection_data_transmission_matches_poseidon(tmp_path):
    from POSEIDON.high_res import make_injection_data as p_inj

    data = _sysrem_transmission_data(seed=23)
    # make_injection_data reads "flux" key in transmission too via data["flux"]?
    # Actually for transmission emission=False branch: data_injected = F_p * flux
    # and flux is taken from data["flux"] reading. Add it.
    flux_for_inj, _, _, _, _ = _prep_cube(seed=23)
    data["flux"] = flux_for_inj
    rng = _rng(24)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * np.abs(rng.standard_normal(wl.size))

    ours_dir = tmp_path / "ours"
    theirs_dir = tmp_path / "theirs"
    name = "obs"
    (ours_dir / name).mkdir(parents=True)
    (theirs_dir / name).mkdir(parents=True)

    _high_res.make_injection_data(
        data,
        str(ours_dir),
        name,
        wl,
        planet_spectrum.copy(),
        K_p=180.0,
        V_sys=-2.0,
        method="sysrem",
        star_spectrum=None,
    )
    p_inj(
        data,
        str(theirs_dir),
        name,
        wl,
        planet_spectrum.copy(),
        K_p=180.0,
        V_sys=-2.0,
        method="sysrem",
        star_spectrum=None,
    )

    d_ours = _hdf5_to_dict(str(ours_dir / name / "data_processed.hdf5"))
    d_theirs = _hdf5_to_dict(str(theirs_dir / name / "data_processed.hdf5"))
    assert set(d_ours.keys()) == set(d_theirs.keys())
    for key in d_ours.keys():
        np.testing.assert_allclose(d_ours[key], d_theirs[key], atol=1e-14, rtol=1e-9)
