"""MacMad17 + Mie cloud / haze parameter unpacking and grid interpolation.

Mirrors `POSEIDON/POSEIDON/parameters.py:unpack_cloud_params` (MacMad17
+ Mie branches), `POSEIDON/POSEIDON/clouds.py:interpolate_sigma_Mie_grid`,
and `POSEIDON/POSEIDON/clouds.py:Mie_cloud`.

v1-B: `interpolate_sigma_Mie_grid` numeric kernel uses
`_jax_interpolate.regular_grid_interp_linear`; the surrounding
string-dispatch and dict-typed return preserve POSEIDON's API.
`unpack_MacMad17_cloud_params` / `unpack_Mie_cloud_params` are
JAX-pure: parameter-name lookups happen at setup time and produce
static integer offsets; the resulting `clouds_in` arithmetic uses
`jnp` so the unpacked values flow through jit.

Iceberg is dropped by design (POSEIDON open-source does not implement);
eddysed integration with the runtime forward model is the follow-up
phase 0.5.14.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402  (setup-time name lookups)

from jaxposeidon._aerosol_db_loader import load_aerosol_grid  # noqa: E402, F401
from jaxposeidon._jax_interpolate import (  # noqa: E402
    regular_grid_interp_linear,
)

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
    """POSEIDON `clouds.py:60-63`.

    Setup-time helper: invoked with static `array` and traced `value`
    only when called from a pure numpy path (the `Mie_cloud` runtime
    wraps this in a jit-friendly call via `jnp.argmin`).
    """
    array = np.asarray(array)
    return int((np.abs(array - value)).argmin())


def _find_nearest_jax(array, value):
    return jnp.argmin(jnp.abs(jnp.asarray(array) - value))


def unpack_MacMad17_cloud_params(
    *, clouds_in, cloud_param_names, cloud_type, cloud_dim
):
    """v0 MacMad17 cloud parameter unpacking.

    Args:
        clouds_in: array of MacMad17 cloud-parameter values (in the order
            given by `cloud_param_names`).
        cloud_param_names: numpy array of POSEIDON cloud parameter names
            for this configuration.
        cloud_type: 'deck' / 'haze' / 'deck_haze'.
        cloud_dim: 1 or 2.

    Returns:
        dict of unpacked parameters (`a`, `gamma`, `P_cloud`,
        `kappa_cloud_0`, `f_cloud`, `phi_0`, `theta_0`,
        `enable_haze`, `enable_deck`).
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
    clouds_in = jnp.asarray(clouds_in, dtype=jnp.float64)
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

    return dict(
        a=float(a) if not hasattr(a, "shape") or a.shape == () else a,
        gamma=float(gamma)
        if not hasattr(gamma, "shape") or gamma.shape == ()
        else gamma,
        P_cloud=float(P_cloud)
        if not hasattr(P_cloud, "shape") or P_cloud.shape == ()
        else P_cloud,
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
    """
    sigma_Mie_grid = aerosol_grid["sigma_Mie_grid"]
    r_m_grid = np.asarray(aerosol_grid["r_m_grid"])
    wl_grid = np.asarray(aerosol_grid["wl_grid"])

    aerosol_species_str = isinstance(aerosol_species, str)
    if not aerosol_species_str:
        aerosol_species = np.array(aerosol_species)

    def not_valid(params, grid):
        params = np.asarray(params)
        return (np.max(params) < grid[0]) or (np.min(params) > grid[-1])

    if not_valid(wl, wl_grid):
        raise Exception(
            "Requested wavelength range is out of the grid bounds (0.2 to 30 um)."
        )
    if not_valid(r_m_array, r_m_grid):
        raise Exception(
            "Requested mean particle size is out of the grid bounds. (0.001 to 10 um)"
        )

    eff_axis = np.array([0.0, 1.0, 2.0])
    wl_j = jnp.asarray(wl, dtype=jnp.float64)
    if jnp.ndim(wl_j) == 0:
        wl_j = wl_j[None]

    def interpolate(species):
        if aerosol_species_str:
            if species != aerosol_species:
                raise KeyError(species)
            q = 0
            r_m_lookup = (
                jnp.asarray(r_m_array)
                if jnp.ndim(jnp.asarray(r_m_array)) == 0
                else jnp.asarray(r_m_array)[0]
            )
        else:
            q = int(np.where(aerosol_species == species)[0][0])
            r_m_lookup = jnp.asarray(r_m_array)[q]
        values = jnp.asarray(sigma_Mie_grid[q, :, :, :], dtype=jnp.float64)
        r_m_b = jnp.broadcast_to(r_m_lookup, wl_j.shape)
        results = []
        for eff_idx in (0, 1, 2):
            eff_b = jnp.full_like(wl_j, float(eff_idx))
            qp = jnp.stack([eff_b, r_m_b, wl_j], axis=-1)
            results.append(
                regular_grid_interp_linear((eff_axis, r_m_grid, wl_grid), values, qp)
            )
        return results

    if not return_dict:
        if aerosol_species_str:
            return interpolate(aerosol_species)
        return jnp.stack(
            [jnp.stack(interpolate(species), axis=0) for species in aerosol_species],
            axis=0,
        )

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
    the cloud_dim=1 patchy-free configuration. Returns the slice of
    POSEIDON's `unpack_cloud_params` return tuple that drives
    `Mie_cloud` + the runtime extinction path.
    """
    if cloud_type not in _MIE_SUPPORTED_CLOUD_TYPES:
        raise NotImplementedError(
            f"Mie cloud_type={cloud_type!r}: v0.5.12b supports only "
            f"{sorted(_MIE_SUPPORTED_CLOUD_TYPES)}"
        )
    if cloud_dim != 1:
        raise NotImplementedError(
            f"Mie cloud_dim={cloud_dim}: v0.5.12b supports only cloud_dim=1"
        )
    aerosol_species_list = list(aerosol_species)
    if aerosol_species_list in (["free"], ["file_read"]):
        raise NotImplementedError(
            "Mie aerosol_species=['free'|'file_read']: LX-MIE / file_read "
            "paths are the Phase 0.5.12c follow-up"
        )

    cloud_param_names = np.asarray(cloud_param_names)
    clouds_in = jnp.asarray(clouds_in, dtype=jnp.float64)

    a, gamma = 1.0, -4.0
    kappa_cloud_0 = 1.0e250

    f_cloud, phi_0, theta_0 = 1.0, -90.0, -90.0

    r_m = jnp.stack(
        [
            10.0
            ** clouds_in[int(np.where(cloud_param_names == f"log_r_m_{aer}")[0][0])]
            for aer in aerosol_species_list
        ]
    )

    if cloud_type == "uniform_X":
        log_X_Mie = jnp.stack(
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
        P_cloud = 10.0 ** jnp.stack(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        P_slab_bottom = 10.0 ** (
            jnp.log10(P_cloud)
            + jnp.stack(
                [
                    clouds_in[
                        int(np.where(cloud_param_names == f"Delta_log_P_{aer}")[0][0])
                    ]
                    for aer in aerosol_species_list
                ]
            )
        )
        log_X_Mie = jnp.stack(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "fuzzy_deck":
        P_cloud = 10.0 ** jnp.stack(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_deck_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = jnp.stack(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_n_max_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        fractional_scale_height = jnp.stack(
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
        log_X_Mie = jnp.stack(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        P_cloud = jnp.stack([P_deck])
        P_slab_bottom = 100.0
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "opaque_deck_plus_slab":
        P_deck = (
            10.0
            ** clouds_in[int(np.where(cloud_param_names == "log_P_top_deck")[0][0])]
        )
        P_slab = 10.0 ** jnp.stack(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in aerosol_species_list
            ]
        )
        P_cloud = jnp.concatenate([P_deck[None], P_slab])
        P_slab_bottom = 10.0 ** (
            jnp.log10(P_slab)
            + jnp.stack(
                [
                    clouds_in[
                        int(np.where(cloud_param_names == f"Delta_log_P_{aer}")[0][0])
                    ]
                    for aer in aerosol_species_list
                ]
            )
        )
        log_X_Mie = jnp.stack(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in aerosol_species_list
            ]
        )
        log_n_max = 0
        fractional_scale_height = 0

    elif cloud_type == "fuzzy_deck_plus_slab":
        deck_aer = aerosol_species_list[0]
        slab_aerosols = aerosol_species_list[1:]
        P_deck = (
            10.0
            ** clouds_in[
                int(np.where(cloud_param_names == f"log_P_top_deck_{deck_aer}")[0][0])
            ]
        )
        P_slab = 10.0 ** jnp.stack(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_P_top_slab_{aer}")[0][0])
                ]
                for aer in slab_aerosols
            ]
        )
        P_cloud = jnp.concatenate([P_deck[None], P_slab])
        log_n_max = jnp.stack(
            [
                clouds_in[
                    int(np.where(cloud_param_names == f"log_n_max_{deck_aer}")[0][0])
                ]
            ]
        )
        fractional_scale_height = jnp.stack(
            [clouds_in[int(np.where(cloud_param_names == f"f_{deck_aer}")[0][0])]]
        )
        log_X_Mie = jnp.stack(
            [
                clouds_in[int(np.where(cloud_param_names == f"log_X_{aer}")[0][0])]
                for aer in slab_aerosols
            ]
        )
        P_slab_bottom = 10.0 ** (
            jnp.log10(P_slab)
            + jnp.stack(
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
            jnp.log10(P_cloud)
            + clouds_in[int(np.where(cloud_param_names == "Delta_log_P")[0][0])]
        )
        log_X_Mie = jnp.stack(
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
    path with fixed lognormal logwidth 0.5.
    """
    P = jnp.asarray(P, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    H = jnp.asarray(H, dtype=jnp.float64)
    n = jnp.asarray(n, dtype=jnp.float64)

    n_aerosol_array = []

    for q in range(len(r_m)):
        if cloud_type == "fuzzy_deck":
            P_cloud_index = _find_nearest(np.asarray(P), float(P_cloud[q]))
            cloud_top_height = r[P_cloud_index]
            h = r[P_cloud_index:] - cloud_top_height
            tail = (10 ** log_n_max[q]) * jnp.exp(
                -h / (fractional_scale_height[q] * H[P_cloud_index:])
            )
            head_shape = (P_cloud_index,) + r.shape[1:]
            head = jnp.full(head_shape, 1.0e250, dtype=jnp.float64)
            n_aerosol = jnp.concatenate([head, tail], axis=0)
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "slab":
            P_arr = np.asarray(P)
            P_cloud_index_top = _find_nearest(P_arr, float(P_cloud[q]))
            P_cloud_index_bttm = _find_nearest(P_arr, float(P_cloud_bottom[q]))
            slab_mask = (jnp.arange(P.shape[0]) >= P_cloud_index_bttm) & (
                jnp.arange(P.shape[0]) < P_cloud_index_top
            )
            mask_b = slab_mask.reshape((-1,) + (1,) * (r.ndim - 1))
            n_aerosol = jnp.where(
                mask_b, n * jnp.float_power(10, log_X_Mie[q]), jnp.zeros_like(r)
            )
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "fuzzy_deck_plus_slab":
            if q == 0:
                P_cloud_index = _find_nearest(np.asarray(P), float(P_cloud[q]))
                cloud_top_height = r[P_cloud_index]
                h = r[P_cloud_index:] - cloud_top_height
                tail = (10 ** log_n_max[q]) * jnp.exp(
                    -h / (fractional_scale_height[q] * H[P_cloud_index:])
                )
                head_shape = (P_cloud_index,) + r.shape[1:]
                head = jnp.full(head_shape, 1.0e250, dtype=jnp.float64)
                n_aerosol = jnp.concatenate([head, tail], axis=0)
                n_aerosol_array.append(n_aerosol)
            else:
                P_arr = np.asarray(P)
                P_cloud_index_top = _find_nearest(P_arr, float(P_cloud[q]))
                P_cloud_index_bttm = _find_nearest(P_arr, float(P_cloud_bottom[q - 1]))
                slab_mask = (jnp.arange(P.shape[0]) >= P_cloud_index_bttm) & (
                    jnp.arange(P.shape[0]) < P_cloud_index_top
                )
                mask_b = slab_mask.reshape((-1,) + (1,) * (r.ndim - 1))
                n_aerosol = jnp.where(
                    mask_b,
                    n * jnp.float_power(10, log_X_Mie[q - 1]),
                    jnp.zeros_like(r),
                )
                n_aerosol_array.append(n_aerosol)

        elif cloud_type == "opaque_deck_plus_slab":
            if q == 0:
                P_cloud_index = _find_nearest(np.asarray(P), float(P_cloud[0]))
                head_shape = (P_cloud_index,) + r.shape[1:]
                tail_shape = (P.shape[0] - P_cloud_index,) + r.shape[1:]
                head = jnp.full(head_shape, 1.0e250, dtype=jnp.float64)
                tail = jnp.zeros(tail_shape, dtype=jnp.float64)
                n_aerosol = jnp.concatenate([head, tail], axis=0)
                n_aerosol_array.append(n_aerosol)
            P_arr = np.asarray(P)
            P_cloud_index_top = _find_nearest(P_arr, float(P_cloud[q + 1]))
            P_cloud_index_bttm = _find_nearest(P_arr, float(P_cloud_bottom[q]))
            slab_mask = (jnp.arange(P.shape[0]) >= P_cloud_index_bttm) & (
                jnp.arange(P.shape[0]) < P_cloud_index_top
            )
            mask_b = slab_mask.reshape((-1,) + (1,) * (r.ndim - 1))
            n_aerosol = jnp.where(
                mask_b,
                n * jnp.float_power(10, log_X_Mie[q]),
                jnp.zeros_like(r),
            )
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "opaque_deck_plus_uniform_X":
            if q == 0:
                P_cloud_index = _find_nearest(np.asarray(P), float(P_cloud[0]))
                head_shape = (P_cloud_index,) + r.shape[1:]
                tail_shape = (P.shape[0] - P_cloud_index,) + r.shape[1:]
                head = jnp.full(head_shape, 1.0e250, dtype=jnp.float64)
                tail = jnp.zeros(tail_shape, dtype=jnp.float64)
                n_aerosol = jnp.concatenate([head, tail], axis=0)
                n_aerosol_array.append(n_aerosol)
            n_aerosol = n * jnp.float_power(10, log_X_Mie[q])
            n_aerosol_array.append(n_aerosol)

        elif cloud_type == "one_slab":
            P_arr = np.asarray(P)
            P_cloud_index_top = _find_nearest(P_arr, float(P_cloud))
            P_cloud_index_bttm = _find_nearest(P_arr, float(P_cloud_bottom))
            slab_mask = (jnp.arange(P.shape[0]) >= P_cloud_index_bttm) & (
                jnp.arange(P.shape[0]) < P_cloud_index_top
            )
            mask_b = slab_mask.reshape((-1,) + (1,) * (r.ndim - 1))
            n_aerosol = jnp.where(
                mask_b,
                n * jnp.float_power(10, log_X_Mie[q]),
                jnp.zeros_like(r),
            )
            n_aerosol_array.append(n_aerosol)

        else:  # uniform_X
            n_aerosol = n * jnp.float_power(10, log_X_Mie[q])
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
