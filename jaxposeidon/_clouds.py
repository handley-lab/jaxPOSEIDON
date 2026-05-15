"""MacMad17 + Mie cloud / haze parameter unpacking and grid interpolation.

Mirrors `POSEIDON/POSEIDON/parameters.py:unpack_cloud_params` (MacMad17
+ Mie branches), `POSEIDON/POSEIDON/clouds.py:interpolate_sigma_Mie_grid`,
and `POSEIDON/POSEIDON/clouds.py:Mie_cloud` (the aerosol-database
runtime; the LX-MIE / free / file_read branches of `Mie_cloud_free` are
deferred).

Iceberg is dropped by design (POSEIDON open-source does not implement);
eddysed integration with the runtime forward model is the follow-up
phase 0.5.14.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jaxposeidon._aerosol_db_loader import load_aerosol_grid  # noqa: F401


_MIE_SUPPORTED_CLOUD_TYPES = frozenset(
    {
        "uniform_X",
        "slab",
        "fuzzy_deck",
        "opaque_deck_plus_uniform_X",
        "opaque_deck_plus_slab",
        "fuzzy_deck_plus_slab",
        "one_slab",
    }
)


def _find_nearest(array, value):
    """POSEIDON `clouds.py:60-63`."""
    array = np.asarray(array)
    return int((np.abs(array - value)).argmin())


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


def unpack_Mie_cloud_params(
    *, clouds_in, cloud_param_names, cloud_type, cloud_dim, aerosol_species
):
    """Mie-branch unpacking of POSEIDON `parameters.py:2128-2438`.

    Restricted to the aerosol-database path (not `free`/`file_read`) and
    the cloud_dim=1 patchy-free configuration with no shiny/uniaxial/
    biaxial cloud types. Deferred branches raise NotImplementedError.

    Returns the slice of POSEIDON's `unpack_cloud_params` return tuple
    that drives `Mie_cloud` + the runtime extinction path:
        kappa_cloud_0, P_cloud, f_cloud, phi_0, theta_0, a, gamma,
        r_m, log_n_max, fractional_scale_height,
        log_X_Mie, P_slab_bottom.
    """
    if cloud_type not in _MIE_SUPPORTED_CLOUD_TYPES:
        raise NotImplementedError(
            f"Mie cloud_type={cloud_type!r}: v0.5.12b supports only "
            f"{sorted(_MIE_SUPPORTED_CLOUD_TYPES)} (uniaxial/biaxial/shiny "
            "deferred)"
        )
    if cloud_dim != 1:
        raise NotImplementedError(
            f"Mie cloud_dim={cloud_dim}: v0.5.12b supports only cloud_dim=1 "
            "(patchy Mie deferred)"
        )
    aerosol_species_list = list(aerosol_species)
    if aerosol_species_list in (["free"], ["file_read"]):
        raise NotImplementedError(
            "Mie aerosol_species=['free'|'file_read']: LX-MIE / file_read "
            "paths are the Phase 0.5.12c follow-up"
        )

    cloud_param_names = np.asarray(cloud_param_names)
    clouds_in = np.asarray(clouds_in, dtype=float)

    # POSEIDON parameters.py:2139-2143
    a, gamma = 1.0, -4.0
    kappa_cloud_0 = 1.0e250

    # POSEIDON parameters.py:2150-2152 (cloud_dim == 1)
    f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0

    # POSEIDON parameters.py:2186-2187 (log_r_m_std_dev fixed at 0.5).

    # POSEIDON parameters.py:2292+
    # r_m: one per aerosol species, in species order.
    r_m = np.array(
        [
            10.0
            ** clouds_in[int(np.where(cloud_param_names == f"log_r_m_{aer}")[0][0])]
            for aer in aerosol_species_list
        ]
    )

    if cloud_type == "uniform_X":
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        P_cloud = 100.0
        P_slab_bottom = 100.0
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "slab":
        P_cloud = 10.0 ** np.array(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        P_slab_bottom = 10.0 ** (
            np.log10(P_cloud)
            + np.array(
                [
                    clouds_in[
                        int(np.where(cloud_param_names == f"Delta_log_P_{aer}")[0][0])
                    ]
                    for aer in aerosol_species_list
                ]
            )
        )
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "fuzzy_deck":
        P_cloud = 10.0 ** np.array(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_deck_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_n_max_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        fractional_scale_height = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"f_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_X_Mie = 100
        P_slab_bottom = 100.0

    elif cloud_type == "opaque_deck_plus_uniform_X":
        P_deck = (
            10.0
            ** clouds_in[int(np.where(cloud_param_names == "log_P_top_deck")[0][0])]
        )
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        P_cloud = np.array([P_deck])
        P_slab_bottom = 100.0
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "opaque_deck_plus_slab":
        P_deck = (
            10.0
            ** clouds_in[int(np.where(cloud_param_names == "log_P_top_deck")[0][0])]
        )
        P_slab = 10.0 ** np.array(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        P_cloud = np.concatenate(([P_deck], P_slab))
        P_slab_bottom = 10.0 ** (
            np.log10(P_slab)
            + np.array(
                [
                    clouds_in[
                        int(np.where(cloud_param_names == f"Delta_log_P_{aer}")[0][0])
                    ]
                    for aer in aerosol_species_list
                ]
            )
        )
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "fuzzy_deck_plus_slab":
        # First aerosol is the deck; subsequent are slabs.
        deck_aer = aerosol_species_list[0]
        slab_aerosols = aerosol_species_list[1:]
        P_deck = (
            10.0
            ** clouds_in[
                int(np.where(cloud_param_names == f"log_P_top_deck_{deck_aer}")[0][0])
            ]
        )
        P_slab = 10.0 ** np.array(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in slab_aerosols
            ]
        )
        P_cloud = np.concatenate(([P_deck], P_slab))
        log_n_max = np.array(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_n_max_{deck_aer}")[0][0])
                ]
            ]
        )
        fractional_scale_height = np.array(
            [clouds_in[int(np.where(cloud_param_names == f"f_{deck_aer}")[0][0])]]
        )
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in slab_aerosols
            ]
        )
        P_slab_bottom = 10.0 ** (
            np.log10(P_slab)
            + np.array(
                [
                    clouds_in[
                        int(np.where(cloud_param_names == f"Delta_log_P_{aer}")[0][0])
                    ]
                    for aer in slab_aerosols
                ]
            )
        )

    elif cloud_type == "one_slab":
        P_cloud = (
            10.0
            ** clouds_in[int(np.where(cloud_param_names == "log_P_top_slab")[0][0])]
        )
        P_slab_bottom = 10.0 ** (
            np.log10(P_cloud)
            + clouds_in[int(np.where(cloud_param_names == "Delta_log_P")[0][0])]
        )
        log_X_Mie = np.array(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = 0
        fractional_scale_height = 0

    return dict(
        kappa_cloud_0=kappa_cloud_0,
        P_cloud=P_cloud,
        f_cloud=f_cloud,
        phi_0=phi_0,
        theta_0=theta_0,
        a=a,
        gamma=gamma,
        r_m=r_m,
        log_n_max=log_n_max,
        fractional_scale_height=fractional_scale_height,
        log_X_Mie=log_X_Mie,
        P_slab_bottom=P_slab_bottom,
    )


def Mie_cloud(
    P,
    wl,
    r,
    H,
    n,
    r_m,
    aerosol_species,
    cloud_type,
    aerosol_grid,
    P_cloud=0,
    log_n_max=0,
    fractional_scale_height=0,
    log_X_Mie=0,
    P_cloud_bottom=0,
):
    """Number-density + cross-section aerosol assembly.

    Port of POSEIDON `clouds.py:1786-2056` for the aerosol-database
    path with fixed lognormal logwidth 0.5. Returns
    (n_aerosol_array, sigma_ext_cld_array, g_cld_array, w_cld_array)
    each as a list (length = N_aerosols, with an extra leading slot for
    opaque_deck_plus_* cloud types).
    """
    n_aerosol_array = []

    for q in range(len(r_m)):
        if cloud_type == "fuzzy_deck":
            n_aerosol = np.zeros_like(r)
            P_cloud_index = _find_nearest(P, P_cloud[q])
            cloud_top_height = r[P_cloud_index]
            h = r[P_cloud_index:] - cloud_top_height
            n_aerosol[:P_cloud_index] = 1.0e250
            n_aerosol[P_cloud_index:] = (10 ** log_n_max[q]) * np.exp(
                -h / (fractional_scale_height[q] * H[P_cloud_index:])
            )
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "slab":
            n_aerosol = np.zeros_like(r)
            P_cloud_index_top = _find_nearest(P, P_cloud[q])
            P_cloud_index_bttm = _find_nearest(P, P_cloud_bottom[q])
            n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = n[
                P_cloud_index_bttm:P_cloud_index_top
            ] * np.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "fuzzy_deck_plus_slab":
            if q == 0:
                n_aerosol = np.zeros_like(r)
                P_cloud_index = _find_nearest(P, P_cloud[q])
                cloud_top_height = r[P_cloud_index]
                h = r[P_cloud_index:] - cloud_top_height
                n_aerosol[:P_cloud_index] = 1.0e250
                n_aerosol[P_cloud_index:] = (10 ** log_n_max[q]) * np.exp(
                    -h / (fractional_scale_height[q] * H[P_cloud_index:])
                )
                n_aerosol_array.append(n_aerosol)
            else:
                n_aerosol = np.zeros_like(r)
                P_cloud_index_top = _find_nearest(P, P_cloud[q])
                P_cloud_index_bttm = _find_nearest(P, P_cloud_bottom[q - 1])
                n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = n[
                    P_cloud_index_bttm:P_cloud_index_top
                ] * np.float_power(10, log_X_Mie[q - 1])
                n_aerosol_array.append(n_aerosol)

        elif cloud_type == "opaque_deck_plus_slab":
            if q == 0:
                n_aerosol = np.zeros_like(r)
                P_cloud_index = _find_nearest(P, P_cloud[0])
                n_aerosol[:P_cloud_index] = 1.0e250
                n_aerosol_array.append(n_aerosol)
            n_aerosol = np.zeros_like(r)
            P_cloud_index_top = _find_nearest(P, P_cloud[q + 1])
            P_cloud_index_bttm = _find_nearest(P, P_cloud_bottom[q])
            n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = n[
                P_cloud_index_bttm:P_cloud_index_top
            ] * np.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "opaque_deck_plus_uniform_X":
            if q == 0:
                n_aerosol = np.zeros_like(r)
                P_cloud_index = _find_nearest(P, P_cloud[0])
                n_aerosol[:P_cloud_index] = 1.0e250
                n_aerosol_array.append(n_aerosol)
            n_aerosol = np.zeros_like(r)
            n_aerosol = n * np.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "one_slab":
            n_aerosol = np.zeros_like(r)
            P_cloud_index_top = _find_nearest(P, P_cloud)
            P_cloud_index_bttm = _find_nearest(P, P_cloud_bottom)
            n_aerosol[P_cloud_index_bttm:P_cloud_index_top] = n[
                P_cloud_index_bttm:P_cloud_index_top
            ] * np.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        else:  # uniform_X
            n_aerosol = np.zeros_like(r)
            n_aerosol = n * np.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

    sigma_Mie_interp_dict = interpolate_sigma_Mie_grid(
        aerosol_grid, wl, r_m, aerosol_species, return_dict=True
    )

    sigma_ext_cld_array = []
    g_cld_array = []
    w_cld_array = []
    for aerosol in aerosol_species:
        sigma_ext_cld_array.append(sigma_Mie_interp_dict[aerosol]["eff_ext"])
        g_cld_array.append(sigma_Mie_interp_dict[aerosol]["eff_g"])
        w_cld_array.append(sigma_Mie_interp_dict[aerosol]["eff_w"])

    return n_aerosol_array, sigma_ext_cld_array, g_cld_array, w_cld_array
