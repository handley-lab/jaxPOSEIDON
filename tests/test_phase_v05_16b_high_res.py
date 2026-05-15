"""Phase 0.5.16b1: high-resolution data-prep + Doppler + CCF surface.

Ports POSEIDON `high_res.py:257-285, 319-336, 339-344, 347-404, 440-450,
834-859, 862-882`. The PCA path, the h5py prepare step, and the multi-
likelihood prescriptions are deferred to 0.5.16b2.
"""

import numpy as np
import pytest

from jaxposeidon import _high_res


def _rng(seed=0):
    return np.random.default_rng(seed)


def _synthetic_cube(nord=2, nphi=6, npix=40, seed=0):
    rng = _rng(seed)
    flux = 1.0 + 0.01 * rng.standard_normal((nord, nphi, npix))
    uncertainties = 0.01 * np.ones_like(flux)
    return flux, uncertainties


# ---------------------------------------------------------------------------
# fast_filter
# ---------------------------------------------------------------------------
def test_fast_filter_matches_poseidon():
    from POSEIDON.high_res import fast_filter as p_fast_filter

    flux, unc = _synthetic_cube()
    res_ours, U_ours = _high_res.fast_filter(
        flux.copy(), unc.copy(), niter=3, Print=False
    )
    res_theirs, U_theirs = p_fast_filter(flux.copy(), unc.copy(), niter=3, Print=False)
    np.testing.assert_allclose(res_ours, res_theirs, atol=0, rtol=1e-13)
    np.testing.assert_allclose(U_ours, U_theirs, atol=0, rtol=1e-13)


# ---------------------------------------------------------------------------
# fit_out_transit_spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("spec", ["mean", "median"])
def test_fit_out_transit_spec_matches_poseidon(spec):
    from POSEIDON.high_res import fit_out_transit_spec as p_fit

    flux, _ = _synthetic_cube(nord=3, nphi=8, npix=20)
    transit_weight = np.array([1.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 1.0])
    ours = _high_res.fit_out_transit_spec(
        flux.copy(), transit_weight, spec=spec, Print=False
    )
    theirs = p_fit(flux.copy(), transit_weight, spec=spec, Print=False)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_fit_out_transit_spec_bad_spec_raises():
    """POSEIDON `high_res.py:330-331` raises Exception on unknown spec."""
    from POSEIDON.high_res import fit_out_transit_spec as p_fit

    flux, _ = _synthetic_cube(nord=2, nphi=4, npix=10)
    transit_weight = np.array([1.0, 1.0, 0.0, 1.0])
    with pytest.raises(Exception, match="mean"):
        _high_res.fit_out_transit_spec(flux, transit_weight, spec="mode", Print=False)
    with pytest.raises(Exception, match="mean"):
        p_fit(flux, transit_weight, spec="mode", Print=False)


# ---------------------------------------------------------------------------
# get_RV_range
# ---------------------------------------------------------------------------
def test_get_RV_range_matches_poseidon():
    from POSEIDON.high_res import get_RV_range as p_get

    Kp_range = np.linspace(50, 250, 11)
    Vsys_range = np.linspace(-20, 20, 21)
    phi = np.linspace(-0.05, 0.05, 7)
    np.testing.assert_array_equal(
        _high_res.get_RV_range(Kp_range, Vsys_range, phi),
        p_get(Kp_range, Vsys_range, phi),
    )


# ---------------------------------------------------------------------------
# find_nearest_idx
# ---------------------------------------------------------------------------
def test_find_nearest_idx_matches_poseidon():
    from POSEIDON.high_res import find_nearest_idx as p_find

    arr = np.linspace(0, 10, 51)
    for v in [0.0, 0.123, 5.0, 9.87, 10.0]:
        assert _high_res.find_nearest_idx(arr, v) == p_find(arr, v)


# ---------------------------------------------------------------------------
# cross_correlate
# ---------------------------------------------------------------------------
def _ccf_data(nord, nphi, npix, transit=False, seed=1):
    rng = _rng(seed)
    residuals = 0.001 * rng.standard_normal((nord, nphi, npix))
    uncertainties = 0.01 * np.ones((nord, nphi, npix))
    phi = np.linspace(-0.04, 0.04, nphi)
    # Per-order wavelength grids, monotonically increasing, slightly offset
    wl_grid = np.zeros((nord, npix))
    for o in range(nord):
        wl_grid[o] = np.linspace(1.0 + 0.01 * o, 1.5 + 0.01 * o, npix)
    V_bary = 0.05 * rng.standard_normal(nphi)
    data = {
        "residuals": residuals,
        "uncertainties": uncertainties,
        "phi": phi,
        "wl_grid": wl_grid,
        "V_bary": V_bary,
    }
    if transit:
        # Smooth transit weight: 1 → 0 → 1
        data["transit_weight"] = np.array(
            [1.0, 0.7, 0.2, 0.0, 0.2, 0.7, 1.0][:nphi] + [1.0] * max(0, nphi - 7)
        )[:nphi]
    return data


@pytest.mark.parametrize("transit", [False, True])
def test_cross_correlate_matches_poseidon(transit):
    from POSEIDON.high_res import cross_correlate as p_cc

    nord, nphi, npix = 2, 7, 30
    data = _ccf_data(nord, nphi, npix, transit=transit)
    rng = _rng(7)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    Kp_range = np.linspace(150, 200, 4)
    Vsys_range = np.linspace(-10, 10, 5)
    RV_range = _high_res.get_RV_range(Kp_range, Vsys_range, data["phi"])

    ours = _high_res.cross_correlate(
        Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=False
    )
    theirs = p_cc(
        Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=False
    )
    np.testing.assert_allclose(ours[0], theirs[0], atol=0, rtol=1e-13)
    np.testing.assert_allclose(ours[1], theirs[1], atol=0, rtol=1e-13)


def test_cross_correlate_no_V_bary_matches_poseidon():
    """POSEIDON `high_res.py:357-359` falls back to zeros when V_bary is absent."""
    from POSEIDON.high_res import cross_correlate as p_cc

    nord, nphi, npix = 2, 5, 30
    data = _ccf_data(nord, nphi, npix, transit=False)
    del data["V_bary"]
    rng = _rng(11)
    wl = np.linspace(0.95, 1.6, 400)
    planet_spectrum = 0.001 * rng.standard_normal(wl.size)
    Kp_range = np.linspace(150, 200, 3)
    Vsys_range = np.linspace(-10, 10, 5)
    RV_range = _high_res.get_RV_range(Kp_range, Vsys_range, data["phi"])
    ours = _high_res.cross_correlate(
        Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=False
    )
    theirs = p_cc(
        Kp_range, Vsys_range, RV_range, wl, planet_spectrum, data, Print=False
    )
    np.testing.assert_allclose(ours[0], theirs[0], atol=0, rtol=1e-13)
    np.testing.assert_allclose(ours[1], theirs[1], atol=0, rtol=1e-13)


# ---------------------------------------------------------------------------
# get_rot_kernel
# ---------------------------------------------------------------------------
def test_get_rot_kernel_matches_poseidon():
    from POSEIDON.high_res import get_rot_kernel as p_rk

    wl = np.linspace(1.0, 1.5, 1000)
    # Mix of odd, even, and non-integer W_conv — POSEIDON `high_res.py:847`
    # casts via int(W_conv) and centres asymmetrically when even.
    for V_sin_i, W_conv in [
        (5.0, 21),
        (10.0, 11),
        (3.0, 7),
        (5.0, 20),
        (5.0, 8),
        (5.0, 15.7),
    ]:
        np.testing.assert_allclose(
            _high_res.get_rot_kernel(V_sin_i, wl, W_conv),
            p_rk(V_sin_i, wl, W_conv),
            atol=0,
            rtol=1e-13,
        )


# ---------------------------------------------------------------------------
# remove_outliers
# ---------------------------------------------------------------------------
def test_remove_outliers_matches_poseidon(capsys):
    from POSEIDON.high_res import remove_outliers as p_ro

    rng = _rng(2)
    nord, nphi, npix = 2, 4, 60
    wl_grid = np.zeros((nord, npix))
    for o in range(nord):
        wl_grid[o] = np.linspace(1.0 + 0.01 * o, 1.5 + 0.01 * o, npix)
    flux = 1.0 + 0.001 * rng.standard_normal((nord, nphi, npix))
    # Inject a few outliers
    flux[0, 1, 10] += 0.5
    flux[1, 2, 30] -= 0.5

    ours = _high_res.remove_outliers(wl_grid, flux.copy())
    theirs = p_ro(wl_grid, flux.copy())
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)
