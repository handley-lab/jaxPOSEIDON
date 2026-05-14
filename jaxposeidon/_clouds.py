"""MacMad17 + Mie cloud / haze parameter unpacking and grid interpolation.

Mirrors `POSEIDON/POSEIDON/parameters.py:unpack_cloud_params` (MacMad17
branch) and `POSEIDON/POSEIDON/clouds.py:interpolate_sigma_Mie_grid`.

Iceberg is dropped by design (POSEIDON open-source does not implement);
eddysed integration with the runtime forward model is the follow-up
phase 0.5.14.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jaxposeidon._aerosol_db_loader import load_aerosol_grid  # noqa: F401


def unpack_MacMad17_cloud_params(
    *, clouds_in, cloud_param_names, cloud_type, cloud_dim
):
    """v0 MacMad17 cloud parameter unpacking.

    Args:
        clouds_in: array of MacMad17 cloud-parameter values (in the order
            given by `cloud_param_names`).
        cloud_param_names: numpy array of POSEIDON cloud parameter names
            for this configuration (e.g. ['log_a', 'gamma', 'log_P_cloud',
            'phi_cloud']).
        cloud_type: 'deck' / 'haze' / 'deck_haze'.
        cloud_dim: 1 or 2.

    Returns:
        dict with keys:
            a, gamma, P_cloud, kappa_cloud_0,
            f_cloud, phi_0, theta_0,
            enable_haze, enable_deck.
    """
    if cloud_type not in {"deck", "haze", "deck_haze"}:
        raise NotImplementedError(
            f"cloud_type={cloud_type!r}: v0 supports only "
            "{'deck','haze','deck_haze'} for MacMad17"
        )
    if cloud_dim not in (1, 2):
        raise NotImplementedError(
            f"cloud_dim={cloud_dim}: v0 supports only 1 or 2 for MacMad17"
        )

    cloud_param_names = np.asarray(cloud_param_names)
    enable_haze = 1 if "haze" in cloud_type else 0
    enable_deck = 1 if "deck" in cloud_type else 0

    kappa_cloud_0 = 1.0e250

    if enable_haze == 1:
        a = 10.0 ** clouds_in[int(np.where(cloud_param_names == "log_a")[0][0])]
        gamma = clouds_in[int(np.where(cloud_param_names == "gamma")[0][0])]
    else:
        a, gamma = 1.0, -4.0

    if enable_deck == 1:
        P_cloud = (
            10.0 ** clouds_in[int(np.where(cloud_param_names == "log_P_cloud")[0][0])]
        )
    else:
        P_cloud = 100.0

    if cloud_dim != 1:
        phi_c = clouds_in[int(np.where(cloud_param_names == "phi_cloud")[0][0])]
        phi_0 = 0.0
        f_cloud = phi_c
        theta_0 = -90.0
    else:
        if enable_deck == 1:
            f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0
        else:
            f_cloud, phi_0, theta_0 = 0.0, -90.0, 90.0

    # P_cloud is returned as a scalar matching POSEIDON's
    # `unpack_cloud_params` contract. The caller (compute_spectrum /
    # core.py:1668-1669) is responsible for wrapping it in a length-1
    # ndarray before passing to `extinction(...)` which indexes
    # `P_cloud[0]`.
    return dict(
        a=float(a),
        gamma=float(gamma),
        P_cloud=float(P_cloud),
        kappa_cloud_0=float(kappa_cloud_0),
        f_cloud=float(f_cloud),
        phi_0=float(phi_0),
        theta_0=float(theta_0),
        enable_haze=enable_haze,
        enable_deck=enable_deck,
    )


def interpolate_sigma_Mie_grid(
    aerosol_grid,
    wl,
    r_m_array,
    aerosol_species,
    return_dict=True,
):
    """Interpolate aerosol cross-section grid onto (wl, r_m).

    Bit-equivalent port of POSEIDON `clouds.py:1642-1775` for the
    fixed-logwidth (0.5) case. Returns either a dict keyed by species
    (with `eff_ext`/`eff_g`/`eff_w` subkeys) or a stacked array.

    The free-logwidth path is deferred; pass it explicitly via the
    `aerosol_grid` returned by `load_aerosol_grid` (default).
    """
    sigma_Mie_grid = aerosol_grid["sigma_Mie_grid"]
    r_m_grid = aerosol_grid["r_m_grid"]
    wl_grid = aerosol_grid["wl_grid"]

    aerosol_species_str = isinstance(aerosol_species, str)
    if not aerosol_species_str:
        aerosol_species = np.array(aerosol_species)
    np.seterr(divide="ignore")

    def not_valid(params, grid):
        return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(wl, wl_grid):
        raise Exception(
            "Requested wavelength range is out of the grid bounds (0.2 to 30 um)."
        )
    if not_valid(r_m_array, r_m_grid):
        raise Exception(
            "Requested mean particle size is out of the grid bounds. (0.001 to 10 um)"
        )

    def interpolate(species):
        if aerosol_species_str:
            if species != aerosol_species:
                raise KeyError(species)
            q = 0
            r_m_lookup = r_m_array if np.ndim(r_m_array) == 0 else r_m_array[0]
        else:
            q = np.where(aerosol_species == species)[0][0]
            r_m_lookup = r_m_array[q]
        grid_interp = RegularGridInterpolator(
            ([0, 1, 2], r_m_grid, wl_grid), sigma_Mie_grid[q, :, :, :]
        )
        return [
            grid_interp((0, r_m_lookup, wl)),
            grid_interp((1, r_m_lookup, wl)),
            grid_interp((2, r_m_lookup, wl)),
        ]

    if not return_dict:
        if aerosol_species_str:
            return interpolate(aerosol_species)
        return np.array([interpolate(species) for species in aerosol_species])

    sigma_Mie_interp_dict = {}
    if aerosol_species_str:
        sigma_Mie_interp_dict[aerosol_species] = interpolate(aerosol_species)
        return sigma_Mie_interp_dict
    for species in aerosol_species:
        sigma_Mie_interp_dict[species] = {}
        result = interpolate(species)
        sigma_Mie_interp_dict[species]["eff_ext"] = result[0]
        sigma_Mie_interp_dict[species]["eff_g"] = result[1]
        sigma_Mie_interp_dict[species]["eff_w"] = result[2]
    return sigma_Mie_interp_dict
