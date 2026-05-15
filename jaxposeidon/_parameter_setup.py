"""Parameter setup-only module — extracted from `_parameters.py`.

Setup-only: numpy permitted; **must not be called from inside `jit`**.
Allow-listed by the v1 source-grep gate (see `CLAUDE.md`). Contents:

- `V0_*` configuration whitelist constants (POSEIDON-mirror dispatch).
- `assert_v0_model_config(...)` — POSEIDON kwarg-surface validator
  that raises descriptive `NotImplementedError` for v0-deferred
  configurations.
- `assign_free_params(...)` — string-heavy POSEIDON
  `parameters.py:12-1154` port; produces the parameter ordering and
  cumulative-size table used by `_parameters.split_params(...)` (the
  JAX hot-path companion in `_parameters.py`).

Subsequent v0.5 phases (0.5.6/0.5.7/0.5.9/0.5.10/0.5.11/0.5.12/0.5.14/
0.5.15/0.5.16) lift `NotImplementedError` guards here and extend the
parameter ordering in lock-step with POSEIDON.
"""

import numpy as np

# ---------------------------------------------------------------------------
# v0 configuration whitelist
# ---------------------------------------------------------------------------
V0_PT_PROFILES = frozenset({"isotherm", "Madhu"})
V05_PT_PROFILES_1D = frozenset(
    {
        "isotherm",
        "Madhu",
        "slope",
        "Pelletier",
        "Guillot",
        "Guillot_dayside",
        "Line",
        "gradient",
        "two-gradients",
        "file_read",
    }
)
V0_X_PROFILES = frozenset({"isochem"})
V05_X_PROFILES = frozenset(
    {
        "isochem",
        "gradient",
        "two-gradients",
        "dissociation",
        "lever",
        "file_read",
        "chem_eq",
    }
)
V0_CLOUD_MODELS = frozenset({"cloud-free", "MacMad17"})
V0_CLOUD_TYPES = frozenset({"deck", "haze", "deck_haze"})
V0_CLOUD_DIMS = frozenset({1, 2})
V0_REFERENCE_PARAMETERS = frozenset({"R_p_ref", "P_ref", "R_p_ref+P_ref"})
V0_OFFSETS = frozenset({None, "single_dataset", "two_datasets", "three_datasets"})
V0_ERROR_INFLATIONS = frozenset({None, "Line15", "Piette20", "Line15+Piette20"})


def assert_v0_model_config(
    *,
    PT_profile,
    X_profile,
    cloud_model,
    cloud_dim,
    PT_dim=1,
    X_dim=1,
    Atmosphere_dimension=1,
    cloud_type="deck",
    reference_parameter="R_p_ref",
    object_type="transiting",
    gravity_setting="fixed",
    mass_setting="fixed",
    bulk_species=("H2",),
    stellar_contam=None,
    offsets_applied=None,
    error_inflation=None,
    surface=False,
    surface_model="gray",
    high_res_method=None,
    opaque_Iceberg=False,
    aerosol_species=(),
    species_EM_gradient=(),
    species_DN_gradient=(),
    species_vert_gradient=(),
    TwoD_type=None,
    disable_atmosphere=False,
    sharp_DN_transition=False,
    sharp_EM_transition=False,
    PT_penalty=False,
    lognormal_logwidth_free=False,
):
    """Raise NotImplementedError for any configuration outside the v0 envelope.

    Accepts every primary and secondary tuning knob from POSEIDON's
    `assign_free_params` so the API surface mirrors POSEIDON's. v0
    deferrals raise `NotImplementedError` rather than `TypeError`.
    Tuning kwargs that are inert under the v0 envelope (e.g.
    `log_P_slope_arr` only matters for Pelletier, which is deferred) are
    accepted by `assign_free_params` and silently propagated as no-ops.
    """
    if object_type != "transiting":
        raise NotImplementedError("v0 supports object_type='transiting' only")
    if reference_parameter not in V0_REFERENCE_PARAMETERS:
        raise NotImplementedError(
            f"reference_parameter={reference_parameter!r} not in v0 "
            f"({sorted(V0_REFERENCE_PARAMETERS)})"
        )
    # gravity_setting / mass_setting (Phase 0.5.2b): {"fixed", "free"}.
    # POSEIDON parameters.py:274-275 forbids both free simultaneously.
    if gravity_setting not in ("fixed", "free"):
        raise NotImplementedError(
            f"gravity_setting={gravity_setting!r} not a known POSEIDON option"
        )
    if mass_setting not in ("fixed", "free"):
        raise NotImplementedError(
            f"mass_setting={mass_setting!r} not a known POSEIDON option"
        )
    if gravity_setting == "free" and mass_setting == "free":
        raise Exception("Error: only one of mass or gravity can be a free parameter.")
    if PT_profile not in V05_PT_PROFILES_1D:
        raise NotImplementedError(
            f"PT_profile={PT_profile!r} not in v0.5 set ({sorted(V05_PT_PROFILES_1D)})"
        )
    if X_profile not in V05_X_PROFILES:
        raise NotImplementedError(
            f"X_profile={X_profile!r} not in v0.5 set ({sorted(V05_X_PROFILES)})"
        )
    if cloud_model not in V0_CLOUD_MODELS:
        raise NotImplementedError(
            f"cloud_model={cloud_model!r} not in v0 ({sorted(V0_CLOUD_MODELS)})"
        )
    if cloud_dim not in V0_CLOUD_DIMS:
        raise NotImplementedError(
            f"cloud_dim={cloud_dim!r} not in v0 ({sorted(V0_CLOUD_DIMS)})"
        )
    if cloud_model == "MacMad17" and cloud_type not in V0_CLOUD_TYPES:
        raise NotImplementedError(
            f"cloud_type={cloud_type!r} not in v0 ({sorted(V0_CLOUD_TYPES)})"
        )
    if cloud_model == "cloud-free" and cloud_type != "deck":
        raise NotImplementedError(
            f"cloud_type={cloud_type!r} ignored by cloud-free models in "
            "POSEIDON; v0 requires the inert default cloud_type='deck' so "
            "the accepted API surface matches the documented envelope."
        )
    if PT_dim not in (1, 2, 3):
        raise NotImplementedError(f"PT_dim={PT_dim} not in {{1, 2, 3}}")
    if X_dim not in (1, 2, 3):
        raise NotImplementedError(f"X_dim={X_dim} not in {{1, 2, 3}}")
    if Atmosphere_dimension not in (1, 2, 3):
        raise NotImplementedError(
            f"Atmosphere_dimension={Atmosphere_dimension} not in {{1, 2, 3}}"
        )
    if stellar_contam is not None and stellar_contam not in (
        "one_spot",
        "one_spot_free_log_g",
        "two_spots",
        "two_spots_free_log_g",
    ):
        raise NotImplementedError(
            f"stellar_contam={stellar_contam!r} not a known POSEIDON option"
        )
    if offsets_applied not in V0_OFFSETS:
        raise NotImplementedError(
            f"offsets_applied={offsets_applied!r} not in v0 "
            f"({sorted(o for o in V0_OFFSETS if o is not None) + [None]})"
        )
    if error_inflation not in V0_ERROR_INFLATIONS:
        raise NotImplementedError(f"error_inflation={error_inflation!r} not in v0")
    if surface_model not in ("gray", "constant", "lab_data"):
        raise NotImplementedError(
            f"surface_model={surface_model!r} not a known POSEIDON option"
        )
    # high_res_method is a list/string of method names; POSEIDON validates
    # internally at retrieval time. No setup-layer rejection here.
    if opaque_Iceberg or list(aerosol_species):
        raise NotImplementedError("Iceberg/Mie aerosols are deferred to v1")
    if list(species_EM_gradient) or list(species_DN_gradient):
        raise NotImplementedError(
            "species_EM_gradient / species_DN_gradient require 2D/3D "
            "atmosphere — deferred to Phase 0.5.9"
        )
    if TwoD_type is not None and TwoD_type not in ("D-N", "E-M"):
        raise NotImplementedError(
            f"TwoD_type={TwoD_type!r} not a known POSEIDON option"
        )
    # PT_penalty is only meaningful with PT_profile='Pelletier'; only the
    # additional sigma_s parameter is appended at this layer. The
    # Pelletier+penalty *prior* (spline smoothness penalty) is deferred to
    # Phase 0.5.10.
    if PT_penalty and PT_profile != "Pelletier":
        raise NotImplementedError("PT_penalty only applies to PT_profile='Pelletier'.")
    if lognormal_logwidth_free:
        raise NotImplementedError(
            "lognormal_logwidth_free only applies to Mie aerosols, which "
            "are deferred to v1."
        )


# ---------------------------------------------------------------------------
# assign_free_params — v0 branches
# ---------------------------------------------------------------------------
def assign_free_params(
    *,
    param_species,
    bulk_species=("H2",),
    object_type="transiting",
    PT_profile,
    X_profile,
    cloud_model,
    cloud_type="deck",
    gravity_setting="fixed",
    mass_setting="fixed",
    stellar_contam=None,
    offsets_applied=None,
    error_inflation=None,
    PT_dim=1,
    X_dim=1,
    cloud_dim=1,
    TwoD_type=None,
    TwoD_param_scheme="difference",
    species_EM_gradient=(),
    species_DN_gradient=(),
    species_vert_gradient=(),
    Atmosphere_dimension=1,
    opaque_Iceberg=False,
    surface=False,
    sharp_DN_transition=False,
    sharp_EM_transition=False,
    reference_parameter="R_p_ref",
    disable_atmosphere=False,
    aerosol_species=(),
    log_P_slope_arr=(-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),
    number_P_knots=0,
    PT_penalty=False,
    high_res_method=None,
    alpha_high_res_option="log",
    fix_alpha_high_res=False,
    fix_W_conv_high_res=False,
    fix_beta_high_res=True,
    fix_Delta_phi_high_res=True,
    lognormal_logwidth_free=False,
    surface_components=(),
    surface_model="gray",
    surface_percentage_option="linear",
    thermal=True,
    reflection=False,
):
    """v0 port of POSEIDON.parameters.assign_free_params (parameters.py:12-1154).

    Mirrors the parameter-ordering conventions of POSEIDON exactly for the
    v0 configuration envelope (see `assert_v0_model_config`). Returns the
    same tuple shape as POSEIDON's function:
        (params, physical_params, PT_params, X_params, cloud_params,
         geometry_params, stellar_params, high_res_params, surface_params,
         N_params_cumulative)
    """
    assert_v0_model_config(
        PT_profile=PT_profile,
        X_profile=X_profile,
        cloud_model=cloud_model,
        cloud_dim=cloud_dim,
        cloud_type=cloud_type,
        PT_dim=PT_dim,
        X_dim=X_dim,
        Atmosphere_dimension=Atmosphere_dimension,
        reference_parameter=reference_parameter,
        object_type=object_type,
        gravity_setting=gravity_setting,
        mass_setting=mass_setting,
        bulk_species=bulk_species,
        stellar_contam=stellar_contam,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
        surface=surface,
        surface_model=surface_model,
        high_res_method=high_res_method,
        opaque_Iceberg=opaque_Iceberg,
        aerosol_species=aerosol_species,
        species_EM_gradient=species_EM_gradient,
        species_DN_gradient=species_DN_gradient,
        species_vert_gradient=species_vert_gradient,
        TwoD_type=TwoD_type,
        disable_atmosphere=disable_atmosphere,
        sharp_DN_transition=sharp_DN_transition,
        sharp_EM_transition=sharp_EM_transition,
        PT_penalty=PT_penalty,
        lognormal_logwidth_free=lognormal_logwidth_free,
    )
    # Inert tuning kwargs (only meaningful under deferred branches) are
    # accepted and ignored; their values do not affect v0 parameter ordering.
    _ = (
        TwoD_param_scheme,
        log_P_slope_arr,
        number_P_knots,
        alpha_high_res_option,
        fix_alpha_high_res,
        fix_W_conv_high_res,
        fix_beta_high_res,
        fix_Delta_phi_high_res,
        surface_components,
        surface_percentage_option,
        thermal,
        reflection,
    )

    params = []
    physical_params = []
    PT_params = []
    X_params = []
    cloud_params = []
    geometry_params = []
    stellar_params = []  # always empty in v0
    high_res_params = []  # always empty in v0
    surface_params = []

    # Physical parameters (POSEIDON parameters.py:262-289)
    if reference_parameter == "R_p_ref":
        physical_params += ["R_p_ref"]
    elif reference_parameter == "P_ref":
        physical_params += ["log_P_ref"]
    elif reference_parameter == "R_p_ref+P_ref":
        physical_params += ["R_p_ref", "log_P_ref"]

    # Phase 0.5.2b: free gravity/mass append after the reference parameter(s)
    # in POSEIDON's order (parameters.py:277-281).
    if gravity_setting == "free":
        physical_params += ["log_g"]
    if mass_setting == "free":
        physical_params += ["M_p"]

    N_physical_params = len(physical_params)
    params += physical_params

    # PT parameters (parameters.py:322-349, 1D only)
    if PT_profile == "isotherm":
        PT_params += ["T"]
    elif PT_profile == "gradient":
        PT_params += ["T_high", "T_deep"]
    elif PT_profile == "two-gradients":
        PT_params += ["T_high", "T_mid", "log_P_mid", "T_deep"]
    elif PT_profile == "file_read":
        pass  # No free PT parameters — profile loaded from file
    elif PT_profile == "Madhu":
        PT_params += ["a1", "a2", "log_P1", "log_P2", "log_P3", "T_ref"]
    elif PT_profile == "slope":
        PT_params += ["T_phot_PT"]
        for i in range(len(log_P_slope_arr)):
            PT_params += [f"Delta_T_{i + 1}"]
    elif PT_profile == "Pelletier":
        if number_P_knots < 3:
            raise Exception(
                "number_P_knots must be at least 3. (Captures top, bottom, "
                "middle pressures in log space)"
            )
        for i in range(number_P_knots):
            PT_params += [f"T_{i + 1}"]
        if PT_penalty:
            PT_params += ["sigma_s"]
    elif PT_profile in ("Guillot", "Guillot_dayside"):
        PT_params += ["log_kappa_IR", "log_gamma", "T_int", "T_equ"]
    elif PT_profile == "Line":
        PT_params += [
            "log_kappa_IR",
            "log_gamma",
            "log_gamma_2",
            "alpha_Line",
            "beta_Line",
            "T_int",
        ]
    N_PT_params = len(PT_params)
    params += PT_params

    # X parameters (parameters.py:434-454, 1D)
    if X_profile == "chem_eq":
        X_params += ["C_to_O", "log_Met"]
    elif X_profile != "file_read":
        for species in param_species:
            has_profile = species in species_vert_gradient
            if has_profile and X_profile == "gradient":
                X_params += [f"log_{species}_high", f"log_{species}_deep"]
            elif has_profile and X_profile == "two-gradients":
                X_params += [
                    f"log_{species}_high",
                    f"log_{species}_mid",
                    f"log_P_{species}_mid",
                    f"log_{species}_deep",
                ]
            elif has_profile and X_profile == "lever":
                X_params += [
                    f"log_{species}_iso",
                    f"log_P_{species}",
                    f"Upsilon_{species}",
                ]
            elif has_profile and X_profile == "dissociation":
                if species in ("H2O", "TiO", "VO", "H-", "Na", "K"):
                    X_params += [f"log_{species}_deep"]
                else:
                    X_params += [f"log_{species}"]
            else:
                X_params += [f"log_{species}"]
    N_species_params = len(X_params)
    params += X_params

    # Cloud parameters (parameters.py:719-743, MacMad17)
    if cloud_model == "cloud-free":
        cloud_params = []
    elif cloud_model == "MacMad17":
        if "haze" in cloud_type:
            cloud_params += ["log_a", "gamma"]
        if "deck" in cloud_type:
            cloud_params += ["log_P_cloud"]
        if cloud_dim == 2:
            cloud_params += ["phi_cloud"]
    N_cloud_params = len(cloud_params)
    params += cloud_params

    # Geometry parameters (parameters.py:995-1009)
    if Atmosphere_dimension == 3:
        if not sharp_DN_transition:
            if not sharp_EM_transition:
                geometry_params += ["alpha", "beta"]
            else:
                geometry_params += ["beta"]
        elif not sharp_EM_transition:
            geometry_params += ["alpha"]
    elif Atmosphere_dimension == 2:
        if TwoD_type == "E-M" and not sharp_EM_transition:
            geometry_params += ["alpha"]
        elif TwoD_type == "D-N" and not sharp_DN_transition:
            geometry_params += ["beta"]
    N_geometry_params = len(geometry_params)
    params += geometry_params

    # Stellar parameters (parameters.py:1016-1031)
    if stellar_contam == "one_spot":
        stellar_params += ["f_het", "T_het", "T_phot"]
    elif stellar_contam == "one_spot_free_log_g":
        stellar_params += ["f_het", "T_het", "T_phot", "log_g_het", "log_g_phot"]
    elif stellar_contam == "two_spots":
        stellar_params += ["f_spot", "f_fac", "T_spot", "T_fac", "T_phot"]
    elif stellar_contam == "two_spots_free_log_g":
        stellar_params += [
            "f_spot",
            "f_fac",
            "T_spot",
            "T_fac",
            "T_phot",
            "log_g_spot",
            "log_g_fac",
            "log_g_phot",
        ]
    N_stellar_params = len(stellar_params)
    params += stellar_params

    # Offsets (parameters.py:1035-1047)
    if offsets_applied is None:
        N_offset_params = 0
    elif offsets_applied == "single_dataset":
        params += ["delta_rel"]
        N_offset_params = 1
    elif offsets_applied == "two_datasets":
        params += ["delta_rel_1", "delta_rel_2"]
        N_offset_params = 2
    elif offsets_applied == "three_datasets":
        params += ["delta_rel_1", "delta_rel_2", "delta_rel_3"]
        N_offset_params = 3

    # Error inflation (parameters.py:1051-1064)
    if error_inflation is None:
        N_error_params = 0
    elif error_inflation == "Line15":
        params += ["b"]
        N_error_params = 1
    elif error_inflation == "Piette20":
        params += ["x_tol"]
        N_error_params = 1
    elif error_inflation == "Line15+Piette20":
        params += ["b", "x_tol"]
        N_error_params = 2

    # High-resolution (parameters.py:1068-1090)
    if high_res_method is not None:
        high_res_params += ["K_p", "V_sys"]
        if not fix_W_conv_high_res:
            high_res_params += ["W_conv"]
        if not fix_Delta_phi_high_res:
            high_res_params += ["Delta_phi"]
        if not fix_alpha_high_res:
            if alpha_high_res_option == "linear":
                high_res_params += ["alpha_HR"]
            elif alpha_high_res_option == "log":
                high_res_params += ["log_alpha_HR"]
        if not fix_beta_high_res:
            high_res_params += ["beta_HR"]
    N_high_res_params = len(high_res_params)
    params += high_res_params

    # Surface (parameters.py:1101-1124)
    if not disable_atmosphere:
        if surface:
            surface_params += ["log_P_surf"]
        if surface_model == "constant":
            surface_params += ["albedo_surf"]
        elif surface_model == "lab_data" and len(surface_components) > 1:
            for n in range(len(surface_components)):
                if surface_percentage_option == "linear":
                    surface_params += [f"{surface_components[n]}_percentage"]
                elif surface_percentage_option == "log":
                    surface_params += [f"log_{surface_components[n]}_percentage"]
    N_surface_params = len(surface_params)
    params += surface_params

    # Convert to numpy arrays
    params = np.array(params)
    physical_params = np.array(physical_params)
    PT_params = np.array(PT_params)
    X_params = np.array(X_params)
    cloud_params = np.array(cloud_params)
    geometry_params = np.array(geometry_params)
    stellar_params = np.array(stellar_params)
    high_res_params = np.array(high_res_params)
    surface_params = np.array(surface_params)

    N_params_cumulative = np.cumsum(
        [
            N_physical_params,
            N_PT_params,
            N_species_params,
            N_cloud_params,
            N_geometry_params,
            N_stellar_params,
            N_offset_params,
            N_error_params,
            N_high_res_params,
            N_surface_params,
        ]
    )

    return (
        params,
        physical_params,
        PT_params,
        X_params,
        cloud_params,
        geometry_params,
        stellar_params,
        high_res_params,
        surface_params,
        N_params_cumulative,
    )
