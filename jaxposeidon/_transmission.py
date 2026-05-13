"""TRIDENT transmission radiative transfer — v0 port.

Faithful **numpy** port of POSEIDON's `transmission.py:12-944` for the
v0 envelope:

- 1D background atmosphere (`N_sectors_back == 1`, `N_zones_back == 1`)
- MacMad17 deck + haze, optionally `cloud_dim=2` patchy (handled via
  TRIDENT's sector/zone splitting machinery)
- Beer-Lambert chord transmission via tensor products

**v0 deferral**: the implementation uses numpy + python for-loops with
data-dependent control flow (sector caching, irregular cloud-boundary
splitting) that does not map cleanly onto `jax.numpy` + `lax`. Bit-exact
POSEIDON parity is the v0 goal; full JAX-tracing (jit, vmap, grad) is
v1 work. The intended Phase 10 BlackJAX NSS sampler does not require
JAX-tracing through `TRIDENT` — it can call the numpy implementation
through a `jax.experimental.io_callback` boundary, exactly like
POSEIDON's PyMultiNest integration calls numpy/numba. HMC/VI users
will need the v1 JAX-traced reimplementation, which is a substantive
rewrite of `extend_rad_transfer_grids`, `path_distribution_geometric`,
and `compute_tau_vert` into `lax.cond`/`lax.fori_loop` form.

Source mapping per function:
    zone_boundaries                — transmission.py:12-83
    path_distribution_geometric    — transmission.py:86-285
    extend_rad_transfer_grids      — transmission.py:288-529
    compute_tau_vert               — transmission.py:532-630
    delta_ray_geom                 — transmission.py:633-687
    area_overlap_circles           — transmission.py:690-719
    TRIDENT                        — transmission.py:722-944
"""

import numpy as np

from jaxposeidon._opacity_precompute import prior_index


def area_overlap_circles(d, r_1, r_2):
    """Analytic overlap area of two circles (port of transmission.py:690-719)."""
    d_sq = d ** 2
    r_1_sq = r_1 ** 2
    r_2_sq = r_2 ** 2
    phi_1 = np.arccos((d_sq + r_1_sq - r_2_sq) / (2 * d * r_1))
    phi_2 = np.arccos((d_sq + r_2_sq - r_1_sq) / (2 * d * r_2))
    return (r_1_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1))
            + r_2_sq * (phi_2 - 0.5 * np.sin(2.0 * phi_2)))


def delta_ray_geom(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq):
    """Bit-exact port of `transmission.py:634-687`."""
    delta_ray = np.zeros((N_b, N_phi))
    for j in range(N_phi):
        for i in range(N_b):
            d_ij_sq = (b[i] ** 2 + b_p ** 2 + y_p ** 2
                       + 2.0 * b[i] * (b_p * np.cos(phi_grid[j] - np.pi / 2.0)
                                        + y_p * np.sin(phi_grid[j] - np.pi / 2.0)))
            delta_ray[i, j] = 1.0 if d_ij_sq <= R_s_sq else 0.0
    return delta_ray


def zone_boundaries(N_b, N_sectors, N_zones, b, r_up, k_zone_back,
                    theta_edge_min, theta_edge_max):
    """Port of `transmission.py:13-83`."""
    r_min = np.zeros((N_b, N_sectors, N_zones))
    r_max = np.zeros((N_b, N_sectors, N_zones))
    for k in range(N_zones):
        denom_min = np.cos(theta_edge_min[k])
        denom_max = np.cos(theta_edge_max[k])
        k_in = k_zone_back[k]
        for i in range(N_b):
            r_min_geom = b[i] / (denom_min + 1.0e-250)
            r_max_geom = b[i] / (denom_max + 1.0e-250)
            for j in range(N_sectors):
                r_min[i, j, k] = np.minimum(r_up[-1, j, k_in], r_min_geom)
                if (k == 0) or (k == N_zones - 1):
                    r_max[i, j, k] = r_up[-1, j, k_in]
                else:
                    r_max[i, j, k] = np.minimum(r_up[-1, j, k_in], r_max_geom)
    return r_min, r_max


def extend_rad_transfer_grids(phi_edge, theta_edge, R_s, d, R_max, f_cloud,
                               phi_0, theta_0, N_sectors_back, N_zones_back,
                               enable_deck, N_phi_max=36):
    """Port of `transmission.py:289-529`."""
    phi_edge_N = np.pi / 2.0 + phi_edge[:-1]
    phi_edge_S = (-1.0 * phi_edge_N)[::-1] + 2.0 * np.pi
    phi_edge_back = np.append(phi_edge_N, phi_edge_S)
    theta_edge_back = theta_edge

    theta_0 = (np.pi / 180.0) * theta_0
    theta_edge_all = theta_edge_back
    if (N_zones_back == 1) and (enable_deck == 1):
        theta_edge_all = np.append(theta_edge_all, 0.0)
    if not np.any(theta_edge_all == theta_0):
        theta_edge_all = np.append(theta_edge_all, theta_0)
    theta_edge_all = np.sort(theta_edge_all)
    dtheta_all = np.diff(theta_edge_all)
    theta_all = (-np.pi / 2.0) + np.cumsum(dtheta_all) - (dtheta_all / 2.0)
    theta_grid = theta_all
    N_zones = len(theta_grid)
    cloudy_zones = np.zeros(N_zones).astype(np.int64)
    cloudy_zones[theta_grid >= theta_0] = 1
    k_zone_back = np.zeros(N_zones).astype(np.int64)
    for k in range(N_zones):
        k_zone_back[k] = prior_index(theta_grid[k], theta_edge_back, 0)

    phi_0 = (np.pi / 180.0) * phi_0
    phi_c = 2.0 * np.pi * f_cloud
    phi_0 = np.pi / 2.0 + phi_0
    phi_0 = np.mod(phi_0, 2.0 * np.pi)
    phi_end = np.mod(phi_0 + phi_c, 2.0 * np.pi)

    phi_edge_all = phi_edge_back
    if not np.any(phi_edge_back == phi_0):
        phi_edge_all = np.append(phi_edge_all, phi_0)
    if not np.any(phi_edge_back == phi_end):
        phi_edge_all = np.append(phi_edge_all, phi_end)
    phi_edge_all = np.sort(phi_edge_all)
    dphi_all = np.diff(phi_edge_all)
    phi_all = 0.0 + np.cumsum(dphi_all) - (dphi_all / 2.0)
    N_sectors = len(phi_all)
    cloudy_sectors = np.zeros(N_sectors).astype(np.int64)
    if (phi_0 + phi_c) < 2.0 * np.pi:
        cloudy_sectors[(phi_all >= phi_0) & (phi_all <= phi_end)] = 1
    else:
        cloudy_sectors[(phi_all <= phi_end) | (phi_all >= phi_0)] = 1

    if d <= (R_s - R_max):
        N_phi = N_sectors
        dphi_grid = dphi_all
        phi_grid = phi_all
    else:
        if N_sectors == 1:
            N_phi = N_sectors
            dphi_grid = dphi_all
            phi_grid = phi_all
        else:
            N_phi = N_phi_max
            dphi_0 = (2.0 * np.pi) / N_phi
            dphi_grid = dphi_0 * np.ones(N_phi)
            phi_grid = np.cumsum(dphi_grid) - (dphi_grid / 2.0)

    j_sector = np.zeros(N_phi).astype(np.int64)
    j_sector_back = np.zeros(N_phi).astype(np.int64)
    for j in range(N_phi):
        j_sector_in = prior_index(phi_grid[j], phi_edge_all, 0)
        j_sector_back_in = prior_index(phi_grid[j], phi_edge_back, 0)
        if j_sector_back_in >= N_sectors_back:
            j_sector_back_in = 2 * (N_sectors_back - 1) - j_sector_back_in
        j_sector[j] = j_sector_in
        j_sector_back[j] = j_sector_back_in

    return (phi_grid, dphi_grid, theta_grid, theta_edge_all,
            N_sectors, N_zones, N_phi, j_sector, j_sector_back,
            k_zone_back, cloudy_sectors, cloudy_zones)


def path_distribution_geometric(b, r_up, r_low, dr, i_bot, j_sector_back,
                                 N_layers, N_sectors_back, N_zones_back,
                                 N_zones, N_phi, k_zone_back, theta_edge_all):
    """Port of `transmission.py:87-285`."""
    N_b = b.shape[0]
    Path = np.zeros((N_b, N_phi, N_zones, N_layers))
    r_up_sq = r_up * r_up
    r_low_sq = r_low * r_low
    b_sq = b * b

    if N_zones == 1:
        symmetry_factor = 2.0
    else:
        symmetry_factor = 1.0

    if (N_zones <= 2) and (N_zones == N_zones_back):
        for j in range(N_phi):
            j_sector_last = -1
            j_sector_back_in = j_sector_back[j]
            if j_sector_back_in != j_sector_last:
                for k in range(N_zones):
                    for i in range(N_b):
                        for l in range(i_bot, N_layers):
                            if b[i] < r_up[l, j_sector_back_in, k]:
                                s1 = np.sqrt(r_up_sq[l, j_sector_back_in, k] - b_sq[i])
                                if b[i] > r_low[l, j_sector_back_in, k]:
                                    s2 = 0.0
                                else:
                                    s2 = np.sqrt(r_low_sq[l, j_sector_back_in, k] - b_sq[i])
                                Path[i, j, k, l] = symmetry_factor * (s1 - s2) / dr[l, j_sector_back_in, k]
                            else:
                                Path[i, j, k, l] = 0.0
            else:
                Path[:, j, :, :] = Path[:, j - 1, :, :]
            j_sector_last = j_sector_back_in
    else:
        theta_edge_max = np.delete(theta_edge_all, np.where(theta_edge_all == 0.0)[0])
        theta_edge_min = np.sort(np.append(theta_edge_all[1:-1], 0.0))
        r_min, r_max = zone_boundaries(N_b, N_sectors_back, N_zones, b, r_up,
                                        k_zone_back, theta_edge_min, theta_edge_max)
        r_min_sq = r_min * r_min
        r_max_sq = r_max * r_max
        j_sector_last = -1
        for j in range(N_phi):
            j_sector_back_in = j_sector_back[j]
            if j_sector_back_in != j_sector_last:
                for k in range(N_zones):
                    k_in = k_zone_back[k]
                    for i in range(N_b):
                        for l in range(i_bot, N_layers):
                            if ((r_low[l, j_sector_back_in, k_in] >= r_max[i, j_sector_back_in, k]) or
                                (r_up[l, j_sector_back_in, k_in] <= r_min[i, j_sector_back_in, k]) or
                                (b[i] >= r_max[i, j_sector_back_in, k])):
                                Path[i, j, k, l] = 0.0
                            else:
                                if r_up[l, j_sector_back_in, k_in] >= r_max[i, j_sector_back_in, k]:
                                    s1 = np.sqrt(r_max_sq[i, j_sector_back_in, k] - b_sq[i])
                                    s2 = 0.0
                                else:
                                    s2 = np.sqrt(r_up_sq[l, j_sector_back_in, k_in] - b_sq[i])
                                    s1 = 0.0
                                if r_low[l, j_sector_back_in, k_in] > r_min[i, j_sector_back_in, k]:
                                    s3 = np.sqrt(r_low_sq[l, j_sector_back_in, k_in] - b_sq[i])
                                    s4 = 0.0
                                else:
                                    s4 = np.sqrt(r_min_sq[i, j_sector_back_in, k] - b_sq[i])
                                    s3 = 0.0
                                Path[i, j, k, l] = symmetry_factor * (s1 + s2 - s3 - s4) / dr[l, j_sector_back_in, k_in]
            else:
                Path[:, j, :, :] = Path[:, j - 1, :, :]
            j_sector_last = j_sector_back_in
    return Path


def compute_tau_vert(N_phi, N_layers, N_zones, N_wl, j_sector, j_sector_back,
                     k_zone_back, cloudy_zones, cloudy_sectors, kappa_clear,
                     kappa_cloud, dr):
    """Port of `transmission.py:533-630`."""
    tau_vert = np.zeros((N_layers, N_phi, N_zones, N_wl))
    for j in range(N_phi):
        j_sector_last = -1
        j_sector_in = j_sector[j]
        j_sector_back_in = j_sector_back[j]
        if j_sector_in != j_sector_last:
            for k in range(N_zones):
                k_zone_back_in = k_zone_back[k]
                if cloudy_zones[k] == 1 and cloudy_sectors[j_sector_in] == 1:
                    for q in range(N_wl):
                        tau_vert[:, j, k, q] = (
                            (kappa_clear[:, j_sector_back_in, k_zone_back_in, q]
                             + kappa_cloud[:, j_sector_back_in, k_zone_back_in, q])
                            * dr[:, j_sector_back_in, k_zone_back_in]
                        )
                else:
                    for q in range(N_wl):
                        tau_vert[:, j, k, q] = (
                            kappa_clear[:, j_sector_back_in, k_zone_back_in, q]
                            * dr[:, j_sector_back_in, k_zone_back_in]
                        )
        else:
            tau_vert[:, j, :, :] = tau_vert[:, j - 1, :, :]
        j_sector_last = j_sector_in
    return tau_vert


def TRIDENT(P, r, r_up, r_low, dr, wl, kappa_clear, kappa_cloud,
            enable_deck, enable_haze, b_p, y_p, R_s,
            f_cloud, phi_0, theta_0, phi_edge, theta_edge):
    """Port of `transmission.py:722-944`."""
    d_sq = b_p ** 2 + y_p ** 2
    d = np.sqrt(d_sq)
    N_wl = len(wl)
    R_s_sq = R_s * R_s
    N_layers = len(P)
    i_bot = 0
    j_top = int(np.argmax(r[-1, :, 0]))
    R_max = r_up[-1, j_top, 0]
    R_max_sq = R_max * R_max
    b = r_up[:, j_top, 0]
    db = dr[:, j_top, 0]
    N_b = b.shape[0]
    N_sectors_back = r.shape[1]
    N_zones_back = r.shape[2]

    (phi_grid, dphi_grid, theta_grid, theta_edge_all,
     N_sectors, N_zones, N_phi, j_sector, j_sector_back,
     k_zone_back, cloudy_sectors, cloudy_zones) = extend_rad_transfer_grids(
        phi_edge, theta_edge, R_s, d, R_max, f_cloud, phi_0, theta_0,
        N_sectors_back, N_zones_back, enable_deck,
    )

    if d >= (R_s + R_max):
        return np.zeros(N_wl)
    elif d <= (R_s - R_max):
        A_overlap = np.pi * R_max_sq
    else:
        phi_1 = np.arccos((d_sq + R_max_sq - R_s_sq) / (2 * d * R_max))
        phi_2 = np.arccos((d_sq + R_s_sq - R_max_sq) / (2 * d * R_s))
        A_overlap = (R_max_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1))
                     + R_s_sq * (phi_2 - 0.5 * np.sin(2.0 * phi_2)))

    delta_ray = delta_ray_geom(N_phi, N_b, b, b_p, y_p, phi_grid, R_s_sq)

    if (d > (R_s - R_max)) and (d < (R_s + R_max)) and (N_sectors == 1):
        dA_atm = np.outer(
            area_overlap_circles(d, r_up[:, 0, 0], R_s)
            - area_overlap_circles(d, r_low[:, 0, 0], R_s),
            np.ones_like(dphi_grid),
        )
        dA_atm_overlap = dA_atm
    else:
        dA_atm = np.outer(b * db, dphi_grid)
        dA_atm_overlap = delta_ray * dA_atm

    Path = path_distribution_geometric(
        b, r_up, r_low, dr, i_bot, j_sector_back,
        N_layers, N_sectors_back, N_zones_back, N_zones, N_phi,
        k_zone_back, theta_edge_all,
    )

    tau_vert = compute_tau_vert(
        N_phi, N_layers, N_zones, N_wl, j_sector, j_sector_back,
        k_zone_back, cloudy_zones, cloudy_sectors, kappa_clear, kappa_cloud, dr,
    )

    Trans = np.zeros((N_b, N_phi, N_wl))
    j_sector_last = -1
    for j in range(N_phi):
        j_sector_in = j_sector[j]
        if j_sector_in != j_sector_last:
            Trans[:, j, :] = np.exp(
                -1.0 * np.tensordot(Path[:, j, :, :], tau_vert[:, j, :, :],
                                     axes=([2, 1], [0, 1]))
            )
        else:
            Trans[:, j, :] = Trans[:, j - 1, :]
        j_sector_last = j_sector_in

    A_atm_overlap_eff = np.tensordot(Trans, dA_atm_overlap, axes=([0, 1], [0, 1]))
    transit_depth = (A_overlap - A_atm_overlap_eff) / (np.pi * R_s_sq)
    return transit_depth
