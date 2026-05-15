"""compute_spectrum — v0 transmission orchestrator.

Faithful port of the v0-supported portion of POSEIDON `core.py:1303-2132`
`compute_spectrum(...)`. Wires Phases 1-7 into the public API:

    atmosphere → extinction → TRIDENT → spectrum

Supported envelope:
- spectrum_type in {'transmission', 'transmission_time_average',
  'emission', 'dayside_emission', 'nightside_emission',
  'direct_emission'}; the emission paths cover the no-scattering /
  no-surface case via emission_single_stream. thermal_scattering=True
  and surface=True are the Phase 0.5.13d/e follow-ups.
- opacity_treatment in {'opacity_sampling', 'line_by_line'}.
- device='cpu' only.
- cloud_model in {'cloud-free', 'MacMad17'} only.
- N_sectors/N_zones from TRIDENT geometry (1D or cloud_dim=2 patchy).

Matches POSEIDON's NaN-spectrum rejection sentinel
(`core.py:1370-1374`) for atmospheres outside the fine T-grid or marked
non-physical upstream.
"""

import numpy as np

from jaxposeidon._emission import (
    assign_assumptions_and_compute_single_stream_emission,
    build_surf_reflect,
    determine_photosphere_radii,
    emission_single_stream,
)
from jaxposeidon._lbl import extinction_LBL
from jaxposeidon._opacities import extinction
from jaxposeidon._transmission import TRIDENT


def check_atmosphere_physical(atmosphere, opac):
    """Port of POSEIDON `core.py:1255-1300`."""
    if not bool(atmosphere["is_physical"]):
        return False
    if opac["opacity_treatment"] == "opacity_sampling":
        T = atmosphere["T"]
        T_fine = opac["T_fine"]
        if (np.max(T) > np.max(T_fine)) or (np.min(T) < np.min(T_fine)):
            return False
    return True


_V0_CLOUD_MODELS = {"cloud-free", "MacMad17"}
_V0_CLOUD_TYPES = {"deck", "haze", "deck_haze"}
_V0_CLOUD_DIMS = {1, 2}


def compute_spectrum(
    planet,
    star,
    model,
    atmosphere,
    opac,
    wl,
    spectrum_type="transmission",
    save_spectrum=False,
    disable_continuum=False,
    suppress_print=False,
    Gauss_quad=2,
    use_photosphere_radius=True,
    device="cpu",
    y_p=np.array([0.0]),
    return_albedo=False,
    kappa_contributions=(),
    cloud_properties_contributions=(),
):
    """v0 transmission orchestrator.

    Mirrors POSEIDON `core.py:1303-2132` filtered to the transmission /
    opacity-sampling / cpu / MacMad17 path.
    """
    # --- v0-envelope guards BEFORE any atmosphere-dependent computation ---
    if device != "cpu":
        raise NotImplementedError(
            f"device={device!r}: jaxposeidon v0.5 is CPU/numpy parity only; "
            "GPU is the v1 JAX-trace work."
        )
    if return_albedo:
        raise NotImplementedError(
            "return_albedo=True applies to emission/reflection spectrum_types; "
            "the spectrum_type dispatch wiring is the Phase 0.5.13c follow-up."
        )
    disable_atmosphere = model["disable_atmosphere"]

    if spectrum_type not in (
        "transmission",
        "transmission_time_average",
        "emission",
        "dayside_emission",
        "nightside_emission",
        "direct_emission",
    ):
        raise NotImplementedError(
            f"spectrum_type={spectrum_type!r} not a known POSEIDON option"
        )
    if opac["opacity_treatment"] not in ("opacity_sampling", "line_by_line"):
        raise NotImplementedError(
            f"opacity_treatment={opac['opacity_treatment']!r}: only "
            "'opacity_sampling' and 'line_by_line' are supported."
        )
    is_emission = "emission" in spectrum_type
    if is_emission and (model.get("thermal_scattering") or model.get("reflection")):
        raise NotImplementedError(
            "thermal_scattering / reflection in emission: Phase 0.5.13b Toon "
            "solvers are ported, but their integration with compute_spectrum "
            "is the Phase 0.5.13e follow-up."
        )
    cloud_model = model.get("cloud_model", "cloud-free")
    if cloud_model not in _V0_CLOUD_MODELS:
        raise NotImplementedError(
            f"cloud_model={cloud_model!r} is v1 (v0 supports "
            f"{sorted(_V0_CLOUD_MODELS)})"
        )
    if cloud_model == "MacMad17":
        cloud_type = model.get("cloud_type", "")
        if cloud_type not in _V0_CLOUD_TYPES:
            raise NotImplementedError(
                f"cloud_type={cloud_type!r} is v1 (v0 supports "
                f"{sorted(_V0_CLOUD_TYPES)})"
            )
        cloud_dim = model.get("cloud_dim", 1)
        if cloud_dim not in _V0_CLOUD_DIMS:
            raise NotImplementedError(
                f"cloud_dim={cloud_dim!r} is v1 (v0 supports {sorted(_V0_CLOUD_DIMS)})"
            )

    # --- physical-atmosphere check is the LAST guard before computation ---
    if not disable_atmosphere and not check_atmosphere_physical(atmosphere, opac):
        out = np.empty(len(wl))
        out[:] = np.nan
        return out

    # Unpack planet / atmosphere / model bundles.
    b_p = planet["planet_impact_parameter"]
    R_s = star["R_s"]
    P = atmosphere["P"]
    r = atmosphere["r"]
    r_low = atmosphere["r_low"]
    r_up = atmosphere["r_up"]
    dr = atmosphere["dr"]
    n = atmosphere["n"]
    T = atmosphere["T"]
    X = atmosphere["X"]
    X_active = atmosphere["X_active"]
    X_CIA = atmosphere["X_CIA"]
    X_ff = atmosphere["X_ff"]
    X_bf = atmosphere["X_bf"]
    N_sectors = atmosphere["N_sectors"]
    N_zones = atmosphere["N_zones"]
    phi_edge = atmosphere["phi_edge"]
    theta_edge = atmosphere["theta_edge"]
    a = atmosphere["a"]
    gamma = atmosphere["gamma"]
    P_cloud = atmosphere["P_cloud"]
    kappa_cloud_0 = atmosphere["kappa_cloud_0"]
    f_cloud = atmosphere["f_cloud"]
    phi_cloud_0 = atmosphere["phi_cloud_0"]
    theta_cloud_0 = atmosphere["theta_cloud_0"]
    P_surf = atmosphere["P_surf"]
    albedo_deck = atmosphere["albedo_deck"]
    albedo_surf = atmosphere["albedo_surf"]
    T_surf = atmosphere["T_surf"]
    surface_component_percentages = atmosphere["surface_component_percentages"]
    R_p_ref = atmosphere["R_p_ref"]

    surface = model["surface"]
    surface_model = model["surface_model"]
    surface_components = model["surface_components"]
    surface_component_albedos = model["surface_component_albedos"]
    surface_percentage_apply_to = model["surface_percentage_apply_to"]

    # POSEIDON core.py:1470-1472: renormalize percentages
    if (
        surface
        and len(surface_component_percentages) > 0
        and round(np.sum(surface_component_percentages)) != 1.0
    ):
        surface_component_percentages = surface_component_percentages / np.sum(
            surface_component_percentages
        )

    chemical_species = model["chemical_species"]
    active_species = model["active_species"]
    CIA_pairs = model["CIA_pairs"]
    ff_pairs = model["ff_pairs"]
    bf_species = model["bf_species"]

    enable_haze = 1 if "haze" in model["cloud_type"] else 0
    enable_deck = (
        1
        if ("deck" in model["cloud_type"] and "Mie" not in model["cloud_model"])
        else 0
    )

    # Placeholders for the v0 envelope (no Mie aerosols).
    n_aerosol = np.array([np.zeros_like(r)])
    sigma_ext_cloud = np.array([np.zeros_like(wl)])

    if not isinstance(P_cloud, np.ndarray):
        P_cloud = np.array([P_cloud])

    Rayleigh_stored = opac["Rayleigh_stored"]

    # ----- Phase 4: runtime extinction ---------------------------------------
    if len(kappa_contributions):
        kappa_gas, kappa_Ray, kappa_cloud, _kappa_sep = (
            kappa_contributions[0],
            kappa_contributions[1],
            kappa_contributions[2],
            kappa_contributions[3],
        )
    elif disable_atmosphere:
        N_wl = len(wl)
        kappa_gas = np.zeros((len(P), N_sectors, N_zones, N_wl))
        kappa_Ray = np.zeros_like(kappa_gas)
        kappa_cloud = np.zeros_like(kappa_gas)
    elif opac["opacity_treatment"] == "line_by_line":
        kappa_gas, kappa_Ray, kappa_cloud = extinction_LBL(
            chemical_species,
            active_species,
            CIA_pairs,
            ff_pairs,
            bf_species,
            n,
            T,
            P,
            wl,
            X,
            X_active,
            X_CIA,
            X_ff,
            X_bf,
            a,
            gamma,
            P_cloud,
            kappa_cloud_0,
            Rayleigh_stored,
            enable_haze,
            enable_deck,
            enable_surface=(1 if surface else 0),
            N_sectors=N_sectors,
            N_zones=N_zones,
            P_surf=P_surf,
            opacity_database=opac["opacity_database"],
            disable_continuum=disable_continuum,
            suppress_print=suppress_print,
            database_version=opac["database_version"],
        )
    else:
        sigma_stored = opac["sigma_stored"]
        CIA_stored = opac["CIA_stored"]
        ff_stored = opac["ff_stored"]
        bf_stored = opac["bf_stored"]
        T_fine = opac["T_fine"]
        log_P_fine = opac["log_P_fine"]
        kappa_gas, kappa_Ray, kappa_cloud, _kappa_sep = extinction(
            chemical_species,
            active_species,
            CIA_pairs,
            ff_pairs,
            bf_species,
            n,
            T,
            P,
            wl,
            X,
            X_active,
            X_CIA,
            X_ff,
            X_bf,
            a,
            gamma,
            P_cloud,
            kappa_cloud_0,
            sigma_stored,
            CIA_stored,
            Rayleigh_stored,
            ff_stored,
            bf_stored,
            enable_haze,
            enable_deck,
            enable_surface=(1 if surface else 0),
            N_sectors=N_sectors,
            N_zones=N_zones,
            T_fine=T_fine,
            log_P_fine=log_P_fine,
            P_surf=P_surf,
            enable_Mie=0,
            n_aerosol_array=n_aerosol,
            sigma_Mie_array=sigma_ext_cloud,
        )

    if is_emission:
        if "dayside" in spectrum_type:
            zone_idx = 0
        elif "nightside" in spectrum_type:
            zone_idx = -1
        else:
            zone_idx = 0

        if not disable_atmosphere:
            dz = dr[:, 0, zone_idx]
            T_em = T[:, 0, zone_idx]
            kappa_tot = (
                kappa_gas[:, 0, zone_idx, :]
                + kappa_Ray[:, 0, zone_idx, :]
                + kappa_cloud[:, 0, zone_idx, :]
            )
            dtau_tot = np.ascontiguousarray(kappa_tot * dz.reshape((len(P), 1)))
        else:
            dz = np.array([])
            T_em = np.array([])
            kappa_tot = np.array([])
            dtau_tot = np.array([])

        use_surface_path = surface or albedo_deck != -1 or disable_atmosphere
        if use_surface_path:
            surf_reflect, surf_reflect_array = build_surf_reflect(
                wl,
                surface,
                surface_model,
                albedo_deck,
                albedo_surf,
                surface_components,
                surface_component_albedos,
                surface_component_percentages,
                surface_percentage_apply_to,
            )
            cloud_dim = model.get("cloud_dim", 1)
            aerosol_species = model.get("aerosol_species", [])
            F_p, dtau = assign_assumptions_and_compute_single_stream_emission(
                P,
                T_em,
                dz,
                wl,
                kappa_tot,
                dtau_tot,
                kappa_gas if not disable_atmosphere else np.array([]),
                kappa_Ray if not disable_atmosphere else np.array([]),
                kappa_cloud if not disable_atmosphere else np.array([]),
                np.array([]),
                zone_idx,
                Gauss_quad,
                P_cloud,
                cloud_dim,
                aerosol_species,
                f_cloud,
                albedo_deck,
                disable_atmosphere,
                surface,
                surface_model,
                P_surf,
                T_surf,
                surf_reflect,
                surf_reflect_array,
                surface_component_percentages,
                surface_percentage_apply_to,
            )
        else:
            F_p, dtau = emission_single_stream(T_em, dz, wl, kappa_tot, Gauss_quad)
            dtau = np.flip(dtau, axis=0)

        if use_photosphere_radius and not disable_atmosphere:
            R_p_eff = determine_photosphere_radii(
                np.flip(dtau, axis=0),
                np.flip(r_low[:, 0, zone_idx]),
                wl,
                photosphere_tau=2 / 3,
            )
        elif disable_atmosphere:
            R_p_eff = R_p_ref
        else:
            R_p_eff = planet["planet_radius"]

        d = planet.get("system_distance")
        if d is None:
            if "direct" in spectrum_type:
                raise Exception(
                    "Error: no planet system distance provided. For direct "
                    "spectra, must set system_distance in the planet object."
                )
            d = 1  # cancels in transit-ratio

        if "direct" in spectrum_type:
            spectrum = (R_p_eff / d) ** 2 * F_p
        else:
            F_s = star["F_star"]
            wl_s = star["wl_star"]
            if not np.array_equiv(wl_s, wl):
                raise Exception(
                    "Error: wavelength grid for stellar spectrum does not "
                    "match wavelength grid of planet spectrum. Did you "
                    "forget to provide 'wl' to create_star?"
                )
            F_s_obs = (R_s / d) ** 2 * F_s
            F_p_obs = (R_p_eff / d) ** 2 * F_p
            spectrum = F_p_obs / F_s_obs

        if save_spectrum:
            from jaxposeidon._output import write_spectrum

            write_spectrum(
                planet_name=planet["planet_name"],
                model_name=model["model_name"],
                spectrum=spectrum,
                wl=wl,
            )
        return spectrum

    # Transmission paths (TRIDENT chord integration).
    if spectrum_type == "transmission_time_average":
        N_y = len(y_p)
        spectrum_stored = np.zeros(shape=(N_y, len(wl)))
        for i in range(0, (N_y // 2 + 1)):
            spec_i = TRIDENT(
                P=P,
                r=r,
                r_up=r_up,
                r_low=r_low,
                dr=dr,
                wl=wl,
                kappa_clear=(kappa_gas + kappa_Ray),
                kappa_cloud=kappa_cloud,
                enable_deck=enable_deck,
                enable_haze=enable_haze,
                b_p=b_p,
                y_p=y_p[i],
                R_s=R_s,
                f_cloud=f_cloud,
                phi_0=phi_cloud_0,
                theta_0=theta_cloud_0,
                phi_edge=phi_edge,
                theta_edge=theta_edge,
            )
            spectrum_stored[i, :] = spec_i
            if i != N_y // 2:
                spectrum_stored[N_y - 1 - i, :] = spec_i
        spectrum = 0.5 * (
            np.mean(spectrum_stored[1:-1], axis=0) + np.mean(spectrum_stored, axis=0)
        )
    else:
        spectrum = TRIDENT(
            P=P,
            r=r,
            r_up=r_up,
            r_low=r_low,
            dr=dr,
            wl=wl,
            kappa_clear=(kappa_gas + kappa_Ray),
            kappa_cloud=kappa_cloud,
            enable_deck=enable_deck,
            enable_haze=enable_haze,
            b_p=b_p,
            y_p=y_p[0],
            R_s=R_s,
            f_cloud=f_cloud,
            phi_0=phi_cloud_0,
            theta_0=theta_cloud_0,
            phi_edge=phi_edge,
            theta_edge=theta_edge,
        )

    if save_spectrum:
        from jaxposeidon._output import write_spectrum

        write_spectrum(
            planet_name=planet["planet_name"],
            model_name=model["model_name"],
            spectrum=spectrum,
            wl=wl,
        )

    return spectrum
