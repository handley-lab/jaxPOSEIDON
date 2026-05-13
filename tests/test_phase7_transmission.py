"""Phase 7 TRIDENT parity tests against POSEIDON."""

import numpy as np
import pytest

from jaxposeidon import _transmission


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def test_area_overlap_circles_matches_poseidon():
    from POSEIDON.transmission import area_overlap_circles as p_aoc
    d, r1, r2 = 1.2, 1.0, 0.9
    np.testing.assert_array_equal(
        _transmission.area_overlap_circles(d, r1, r2),
        p_aoc(d, r1, r2),
    )


def test_delta_ray_geom_matches_poseidon():
    from POSEIDON.transmission import delta_ray_geom as p_drg
    N_phi, N_b = 4, 5
    b = np.linspace(0.5e8, 1.5e8, N_b)
    b_p, y_p = 0.0, 0.0
    R_s_sq = (7e8) ** 2
    phi_grid = np.linspace(0.0, 2 * np.pi, N_phi, endpoint=False)
    np.testing.assert_array_equal(
        _transmission.delta_ray_geom(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq),
        p_drg(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq),
    )


def test_zone_boundaries_matches_poseidon():
    from POSEIDON.transmission import zone_boundaries as p_zb
    N_b, N_sectors, N_zones = 6, 1, 2
    b = np.linspace(7.0e7, 8.0e7, N_b)
    r_up = 8.0e7 * np.ones((10, N_sectors, N_zones))
    k_zone_back = np.array([0, 0], dtype=np.int64)
    theta_edge_min = np.array([-np.pi / 2.0, 0.0])
    theta_edge_max = np.array([0.0, np.pi / 2.0])
    ours = _transmission.zone_boundaries(N_b, N_sectors, N_zones, b, r_up,
                                          k_zone_back, theta_edge_min,
                                          theta_edge_max)
    theirs = p_zb(N_b, N_sectors, N_zones, b, r_up, k_zone_back,
                  theta_edge_min, theta_edge_max)
    for a, b_ in zip(ours, theirs):
        np.testing.assert_array_equal(a, b_)


# ---------------------------------------------------------------------------
# TRIDENT — canonical Rayleigh oracle parity
# ---------------------------------------------------------------------------
def _canonical_1D_atmosphere(N_layers=50, N_wl=25):
    """Build a 1D atmosphere mirroring test_TRIDENT.py::test_Rayleigh inputs."""
    rng = np.random.default_rng(0)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    R_p = 7.1492e7
    # Geometric layer arrays: r increasing with altitude.
    r = np.linspace(R_p, R_p * 1.1, N_layers).reshape(N_layers, 1, 1)
    r_up = np.zeros_like(r)
    r_low = np.zeros_like(r)
    dr = np.zeros_like(r)
    r_up[1:-1, 0, 0] = 0.5 * (r[2:, 0, 0] + r[1:-1, 0, 0])
    r_low[1:-1, 0, 0] = 0.5 * (r[1:-1, 0, 0] + r[:-2, 0, 0])
    dr[1:-1, 0, 0] = 0.5 * (r[2:, 0, 0] - r[:-2, 0, 0])
    r_up[0, 0, 0] = 0.5 * (r[1, 0, 0] + r[0, 0, 0])
    r_up[-1, 0, 0] = r[-1, 0, 0] + 0.5 * (r[-1, 0, 0] - r[-2, 0, 0])
    r_low[0, 0, 0] = r[0, 0, 0] - 0.5 * (r[1, 0, 0] - r[0, 0, 0])
    r_low[-1, 0, 0] = 0.5 * (r[-1, 0, 0] + r[-2, 0, 0])
    dr[0, 0, 0] = r[1, 0, 0] - r[0, 0, 0]
    dr[-1, 0, 0] = r[-1, 0, 0] - r[-2, 0, 0]
    wl = np.linspace(1.0, 5.0, N_wl)
    kappa_clear = rng.uniform(0.0, 1e-5, size=(N_layers, 1, 1, N_wl))
    kappa_cloud = np.zeros_like(kappa_clear)
    return dict(
        P=P, r=r, r_up=r_up, r_low=r_low, dr=dr, wl=wl,
        kappa_clear=kappa_clear, kappa_cloud=kappa_cloud,
    )


def _trident_args(atm, **overrides):
    args = dict(
        b_p=0.0, y_p=0.0, R_s=6.96e8,
        f_cloud=0.0, phi_0=-90.0, theta_0=90.0,
        phi_edge=np.array([-np.pi / 2.0, np.pi / 2.0]),
        theta_edge=np.array([-np.pi / 2.0, np.pi / 2.0]),
        enable_deck=0, enable_haze=0,
    )
    args.update(overrides)
    args.update(atm)
    return args


def test_TRIDENT_canonical_rayleigh_1D_matches_poseidon():
    from POSEIDON.transmission import TRIDENT as p_TRIDENT
    atm = _canonical_1D_atmosphere()
    args = _trident_args(atm)
    ours = _transmission.TRIDENT(**args)
    theirs = p_TRIDENT(
        args["P"], args["r"], args["r_up"], args["r_low"], args["dr"],
        args["wl"], args["kappa_clear"], args["kappa_cloud"],
        args["enable_deck"], args["enable_haze"],
        args["b_p"], args["y_p"], args["R_s"],
        args["f_cloud"], args["phi_0"], args["theta_0"],
        args["phi_edge"], args["theta_edge"],
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_TRIDENT_with_cloud_deck_1D_matches_poseidon():
    from POSEIDON.transmission import TRIDENT as p_TRIDENT
    atm = _canonical_1D_atmosphere()
    atm["kappa_cloud"] = 1.0e-5 * np.ones_like(atm["kappa_cloud"])
    args = _trident_args(atm, enable_deck=1)
    ours = _transmission.TRIDENT(**args)
    theirs = p_TRIDENT(
        args["P"], args["r"], args["r_up"], args["r_low"], args["dr"],
        args["wl"], args["kappa_clear"], args["kappa_cloud"],
        args["enable_deck"], args["enable_haze"],
        args["b_p"], args["y_p"], args["R_s"],
        args["f_cloud"], args["phi_0"], args["theta_0"],
        args["phi_edge"], args["theta_edge"],
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)


def test_TRIDENT_patchy_cloud_dim2_matches_poseidon():
    """cloud_dim=2 patchy cloud (f_cloud=0.4, enable_deck=1)."""
    from POSEIDON.transmission import TRIDENT as p_TRIDENT
    atm = _canonical_1D_atmosphere()
    atm["kappa_cloud"] = 5.0e-6 * np.ones_like(atm["kappa_cloud"])
    args = _trident_args(atm, enable_deck=1, f_cloud=0.4,
                          phi_0=-45.0, theta_0=-90.0)
    ours = _transmission.TRIDENT(**args)
    theirs = p_TRIDENT(
        args["P"], args["r"], args["r_up"], args["r_low"], args["dr"],
        args["wl"], args["kappa_clear"], args["kappa_cloud"],
        args["enable_deck"], args["enable_haze"],
        args["b_p"], args["y_p"], args["R_s"],
        args["f_cloud"], args["phi_0"], args["theta_0"],
        args["phi_edge"], args["theta_edge"],
    )
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=0)
