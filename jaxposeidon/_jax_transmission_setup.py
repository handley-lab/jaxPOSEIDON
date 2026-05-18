"""Setup-only geometry orchestrator for TRIDENT.

Allow-listed by the source-grep gate. Runs the data-dependent-shape
geometric prep (POSEIDON `transmission.py:289-529, 87-285, 633-687`)
that depends on cloud-morphology parameters and therefore cannot live
inside a jit boundary. Output is a dict of fixed-shape numpy arrays
that the pure-jnp ``TRIDENT_kernel_jit`` consumes.

This must NOT be called from inside ``jit``. ``compute_spectrum``
invokes it at the numpy/JAX boundary (once per parameter set, before
the jit'd kernel).
"""

import numpy as np

from jaxposeidon import _transmission


def TRIDENT_callback_host(
    P,
    r,
    r_up,
    r_low,
    dr,
    wl,
    kappa_clear,
    kappa_cloud,
    enable_deck,
    enable_haze,
    b_p,
    y_p,
    R_s,
    f_cloud,
    phi_0,
    theta_0,
    phi_edge,
    theta_edge,
):
    """Host-side numpy TRIDENT call for the legacy ``jax.pure_callback``
    wrapper in ``_jax_transmission.TRIDENT_callback``. Runs outside jit.

    Setup-only / allow-listed; not jit-traceable.
    """
    out = _transmission.TRIDENT(
        np.asarray(P),
        np.asarray(r),
        np.asarray(r_up),
        np.asarray(r_low),
        np.asarray(dr),
        np.asarray(wl),
        np.asarray(kappa_clear),
        np.asarray(kappa_cloud),
        int(np.asarray(enable_deck)),
        int(np.asarray(enable_haze)),
        float(np.asarray(b_p)),
        float(np.asarray(y_p)),
        float(np.asarray(R_s)),
        float(np.asarray(f_cloud)),
        float(np.asarray(phi_0)),
        float(np.asarray(theta_0)),
        np.asarray(phi_edge),
        np.asarray(theta_edge),
    )
    return out.astype(np.float64)


def setup_TRIDENT_geometry(
    P,
    r,
    r_up,
    r_low,
    dr,
    wl,
    enable_deck,
    enable_haze,
    b_p,
    y_p,
    R_s,
    f_cloud,
    phi_0,
    theta_0,
    phi_edge,
    theta_edge,
):
    """Numpy geometry orchestrator for TRIDENT.

    Runs ``extend_rad_transfer_grids`` + ``path_distribution_geometric``
    + ``delta_ray_geom`` and assembles the fixed-shape tensors
    ``TRIDENT_kernel_jit`` consumes. Setup-only; runs outside ``jit``.

    Returns a dict with the kernel's geometric inputs as numpy arrays.
    Caller converts to ``jnp`` before passing to the jit kernel.
    """
    d_sq = b_p**2 + y_p**2
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
    N_sectors_back = r.shape[1]
    N_zones_back = r.shape[2]

    (
        phi_grid,
        dphi_grid,
        theta_grid,
        theta_edge_all,
        N_sectors,
        N_zones,
        N_phi,
        j_sector,
        j_sector_back,
        k_zone_back,
        cloudy_sectors,
        cloudy_zones,
    ) = _transmission.extend_rad_transfer_grids(
        phi_edge,
        theta_edge,
        R_s,
        d,
        R_max,
        f_cloud,
        phi_0,
        theta_0,
        N_sectors_back,
        N_zones_back,
        enable_deck,
    )

    if d >= (R_s + R_max):
        return {"geometry_empty": True, "N_wl": N_wl}
    elif d <= (R_s - R_max):
        A_overlap = np.pi * R_max_sq
    else:
        phi_1 = np.arccos((d_sq + R_max_sq - R_s_sq) / (2 * d * R_max))
        phi_2 = np.arccos((d_sq + R_s_sq - R_max_sq) / (2 * d * R_s))
        A_overlap = R_max_sq * (phi_1 - 0.5 * np.sin(2.0 * phi_1)) + R_s_sq * (
            phi_2 - 0.5 * np.sin(2.0 * phi_2)
        )

    delta_ray = _transmission.delta_ray_geom(
        N_phi, b.shape[0], b, b_p, y_p, phi_grid, R_s_sq
    )

    if (d > (R_s - R_max)) and (d < (R_s + R_max)) and (N_sectors == 1):
        dA_atm = np.outer(
            _transmission.area_overlap_circles(d, r_up[:, 0, 0], R_s)
            - _transmission.area_overlap_circles(d, r_low[:, 0, 0], R_s),
            np.ones_like(dphi_grid),
        )
        dA_atm_overlap = dA_atm
    else:
        dA_atm = np.outer(b * db, dphi_grid)
        dA_atm_overlap = delta_ray * dA_atm

    Path = _transmission.path_distribution_geometric(
        b,
        r_up,
        r_low,
        dr,
        i_bot,
        j_sector_back,
        N_layers,
        N_sectors_back,
        N_zones_back,
        N_zones,
        N_phi,
        k_zone_back,
        theta_edge_all,
    )

    return {
        "geometry_empty": False,
        "j_sector": j_sector,
        "j_sector_back": j_sector_back,
        "k_zone_back": k_zone_back,
        "cloudy_zones": cloudy_zones,
        "cloudy_sectors": cloudy_sectors,
        "Path": Path,
        "dA_atm_overlap": dA_atm_overlap,
        "A_overlap": A_overlap,
        "R_s_sq": R_s_sq,
        "N_wl": N_wl,
    }
