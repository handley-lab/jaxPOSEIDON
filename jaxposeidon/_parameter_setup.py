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
V0_PT_PROFILES = {"isotherm", "Madhu"}
V0_X_PROFILES = {"isochem"}
V0_CLOUD_MODELS = {"cloud-free", "MacMad17"}
V0_CLOUD_TYPES = {"deck", "haze", "deck_haze"}
V0_CLOUD_DIMS = {1, 2}
V0_REFERENCE_PARAMETERS = {"R_p_ref", "P_ref", "R_p_ref+P_ref"}
V0_OFFSETS = {None, "single_dataset", "two_datasets", "three_datasets"}
V0_ERROR_INFLATIONS = {None, "Line15", "Piette20", "Line15+Piette20"}


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
    if disable_atmosphere:
        raise NotImplementedError("v0 does not support disable_atmosphere=True")
    if reference_parameter not in V0_REFERENCE_PARAMETERS:
        raise NotImplementedError(
            f"reference_parameter={reference_parameter!r} not in v0 "
            f"({sorted(V0_REFERENCE_PARAMETERS)})"
        )
    if gravity_setting != "fixed":
        raise NotImplementedError("v0 requires gravity_setting='fixed'")
    if mass_setting != "fixed":
        raise NotImplementedError("v0 requires mass_setting='fixed'")
    if "ghost" in bulk_species:
        raise NotImplementedError("v0 does not support 'ghost' bulk species")
    if PT_profile not in V0_PT_PROFILES:
        raise NotImplementedError(
            f"PT_profile={PT_profile!r} not in v0 ({sorted(V0_PT_PROFILES)})"
        )
    if X_profile not in V0_X_PROFILES:
        raise NotImplementedError(
            f"X_profile={X_profile!r} not in v0 ({sorted(V0_X_PROFILES)})"
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
    if PT_dim != 1 or X_dim != 1:
        raise NotImplementedError("v0 supports only PT_dim=1, X_dim=1")
    if Atmosphere_dimension != 1:
        raise NotImplementedError(
            f"Atmosphere_dimension={Atmosphere_dimension} != 1; v0 is 1D only"
        )
    if stellar_contam is not None:
        raise NotImplementedError("Stellar contamination is deferred to v1")
    if offsets_applied not in V0_OFFSETS:
        raise NotImplementedError(
            f"offsets_applied={offsets_applied!r} not in v0 "
            f"({sorted(o for o in V0_OFFSETS if o is not None) + [None]})"
        )
    if error_inflation not in V0_ERROR_INFLATIONS:
        raise NotImplementedError(f"error_inflation={error_inflation!r} not in v0")
    if surface:
        raise NotImplementedError("Surfaces are deferred to v1")
    if surface_model not in ("gray", "constant", "lab_data"):
        raise NotImplementedError(
            f"surface_model={surface_model!r} not a known POSEIDON option"
        )
    if surface_model != "gray":
        raise NotImplementedError("v0 supports only surface_model='gray'")
    if high_res_method is not None:
        raise NotImplementedError("High-resolution mode is deferred to v1")
    if opaque_Iceberg or list(aerosol_species):
        raise NotImplementedError("Iceberg/Mie aerosols are deferred to v1")
    if (
        list(species_EM_gradient)
        or list(species_DN_gradient)
        or list(species_vert_gradient)
    ):
        raise NotImplementedError("v0 forbids per-species chemistry gradients")
    if TwoD_type is not None:
        raise NotImplementedError("v0 forbids TwoD_type")
    # Atmosphere_dimension=1 ⇒ POSEIDON does not insert geometry params
    # regardless of sharp_*_transition values. Still, the v0 envelope is
    # explicitly 1D-only, so we reject non-default sharp-transition flags
    # to avoid silently accepting a 2D/3D-only tuning knob.
    if sharp_DN_transition or sharp_EM_transition:
        raise NotImplementedError(
            "sharp_DN_transition / sharp_EM_transition only apply to 2D/3D "
            "atmospheres, which are deferred to v1."
        )
    # PT_penalty is only meaningful with PT_profile='Pelletier' (deferred);
    # reject if set regardless so a caller cannot pass an obsolete flag.
    if PT_penalty:
        raise NotImplementedError(
            "PT_penalty only applies to PT_profile='Pelletier', which is "
            "deferred to v1."
        )
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

    N_physical_params = len(physical_params)
    params += physical_params

    # PT parameters (parameters.py:322-349, 1D only)
    if PT_profile == "isotherm":
        PT_params += ["T"]
    elif PT_profile == "Madhu":
        PT_params += ["a1", "a2", "log_P1", "log_P2", "log_P3", "T_ref"]
    N_PT_params = len(PT_params)
    params += PT_params

    # X parameters (parameters.py:436-454, 1D isochem only)
    for species in param_species:
        X_params += ["log_" + species]
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

    # Geometry parameters (parameters.py:995-1009, Atmosphere_dimension=1 ⇒ empty)
    N_geometry_params = 0  # 1D atmosphere: no alpha/beta

    # Stellar parameters (parameters.py:1016-1031): always empty in v0
    N_stellar_params = 0

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

    # High-resolution (always empty in v0)
    N_high_res_params = 0

    # Surface (parameters.py:1101-1124, gray surface_model only)
    # surface=False in v0 ⇒ surface_params stays []
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
