"""Phase 1 tests for jaxposeidon._parameters.

Every test compares jaxposeidon's output against POSEIDON's own
parameters.py:assign_free_params / split_params, treating POSEIDON as the
numerical oracle.
"""

import numpy as np
import pytest

from jaxposeidon._parameters import (
    split_params,
    assign_free_params,
    assert_v0_model_config,
    V0_PT_PROFILES,
    V0_X_PROFILES,
    V0_CLOUD_MODELS,
    V0_CLOUD_TYPES,
    V0_CLOUD_DIMS,
    V0_REFERENCE_PARAMETERS,
    V0_OFFSETS,
    V0_ERROR_INFLATIONS,
)


# ---------------------------------------------------------------------------
# split_params
# ---------------------------------------------------------------------------
def test_split_params_matches_poseidon_exactly():
    """jaxposeidon.split_params bit-equivalent to POSEIDON's split_params."""
    from POSEIDON.parameters import split_params as poseidon_split

    rng = np.random.default_rng(0)
    boundaries = np.array([2, 6, 9, 11, 12, 12, 13, 13, 13, 13])
    theta = rng.standard_normal(boundaries[-1]).astype(np.float64)

    ours = split_params(theta, boundaries)
    theirs = poseidon_split(theta, boundaries)

    assert len(ours) == len(theirs) == 10
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)


def test_split_params_handles_empty_groups():
    """Cumulative array where several groups are empty still works."""
    from POSEIDON.parameters import split_params as poseidon_split

    boundaries = np.array([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])
    theta = np.arange(3, dtype=np.float64)
    ours = split_params(theta, boundaries)
    theirs = poseidon_split(theta, boundaries)
    for a, b in zip(ours, theirs):
        np.testing.assert_array_equal(a, b)
    for group in ours[1:]:
        assert group.size == 0


# ---------------------------------------------------------------------------
# v0 whitelist
# ---------------------------------------------------------------------------
def test_v0_whitelist_constants():
    """v0 whitelist matches plan-documented config envelope."""
    assert V0_PT_PROFILES == {"isotherm", "Madhu"}
    assert V0_X_PROFILES == {"isochem"}
    assert V0_CLOUD_MODELS == {"cloud-free", "MacMad17"}
    assert V0_CLOUD_TYPES == {"deck", "haze", "deck_haze"}
    assert V0_CLOUD_DIMS == {1, 2}
    assert V0_REFERENCE_PARAMETERS == {"R_p_ref", "P_ref", "R_p_ref+P_ref"}
    assert V0_OFFSETS == {None, "single_dataset", "two_datasets", "three_datasets"}
    assert V0_ERROR_INFLATIONS == {None, "Line15", "Piette20", "Line15+Piette20"}


def test_v0_config_accepts_canonical_rayleigh():
    """The canonical Rayleigh oracle uses an isotherm PT — must be v0-valid."""
    assert_v0_model_config(
        PT_profile="isotherm", X_profile="isochem",
        cloud_model="cloud-free", cloud_dim=1,
    )


def test_v0_config_accepts_k2_18b_one_offset():
    """K2-18 b paper one-offset case: Madhu PT, isochem, MacMad17, cloud_dim=2."""
    assert_v0_model_config(
        PT_profile="Madhu", X_profile="isochem",
        cloud_model="MacMad17", cloud_dim=2,
        cloud_type="deck_haze",
        reference_parameter="P_ref",
        offsets_applied="single_dataset",
    )


@pytest.mark.parametrize("kwargs,err_substring", [
    (dict(PT_profile="bogus", X_profile="isochem",
          cloud_model="cloud-free", cloud_dim=1), "PT_profile"),
    (dict(PT_profile="Madhu", X_profile="bogus",
          cloud_model="cloud-free", cloud_dim=1), "X_profile"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="Iceberg", cloud_dim=1), "cloud_model"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=3), "cloud_dim"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2, PT_dim=4), "PT_dim"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          stellar_contam="one_spot"), "Stellar"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2, surface=True), "Surface"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          high_res_method="x"), "High-resolution"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          aerosol_species=("ZnS",)), "Iceberg/Mie"),
    # Phase 0.5.2b: gravity_setting/mass_setting='free' now supported.
    # Phase 0.5.7: species_vert_gradient + ghost bulk now supported.
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          object_type="directly_imaged"), "transiting"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          disable_atmosphere=True), "disable_atmosphere"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          reference_parameter="invalid"), "reference_parameter"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          X_dim=4), "X_dim"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          cloud_type="shiny_deck"), "cloud_type"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          offsets_applied="four_datasets"), "offsets_applied"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          error_inflation="Custom"), "error_inflation"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          opaque_Iceberg=True), "Iceberg/Mie"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          TwoD_type="bogus"), "TwoD_type"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          Atmosphere_dimension=4), "Atmosphere_dimension"),
    # PT_penalty with non-Pelletier profile is still rejected.
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          PT_penalty=True), "Pelletier"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          lognormal_logwidth_free=True), "lognormal_logwidth_free"),
    (dict(PT_profile="Madhu", X_profile="isochem",
          cloud_model="MacMad17", cloud_dim=2,
          surface_model="constant"), "surface_model='gray'"),
    (dict(PT_profile="isotherm", X_profile="isochem",
          cloud_model="cloud-free", cloud_dim=1,
          cloud_type="haze"), "cloud-free"),
    (dict(PT_profile="isotherm", X_profile="isochem",
          cloud_model="cloud-free", cloud_dim=1,
          cloud_type="nonsense"), "cloud-free"),
])
def test_v0_config_rejects_out_of_scope(kwargs, err_substring):
    """Non-v0 configurations raise NotImplementedError with descriptive message."""
    with pytest.raises(NotImplementedError, match=err_substring):
        assert_v0_model_config(**kwargs)


def test_unsupported_kwargs_raise_NotImplementedError_not_TypeError():
    """POSEIDON-side deferred kwargs raise NotImplementedError when actively set."""
    base = dict(
        param_species=["H2O"],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    )
    # Each of these activates a not-yet-ported POSEIDON branch and must NOT TypeError.
    for kw in [{"PT_penalty": True}, {"lognormal_logwidth_free": True},
               {"opaque_Iceberg": True}]:
        with pytest.raises(NotImplementedError):
            assign_free_params(**base, **kw)


def test_inert_tuning_kwargs_silently_accepted():
    """Tuning knobs only relevant under deferred branches are accepted as no-ops."""
    base = dict(
        param_species=["H2O"],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    )
    # These shouldn't change v0 output even at unusual values because the
    # branches they control are deferred.
    out_default = assign_free_params(**base)
    for kw in [{"TwoD_param_scheme": "absolute"},
               {"log_P_slope_arr": (-2.0, -1.0)},
               {"number_P_knots": 5},
               {"alpha_high_res_option": "linear"},
               {"fix_alpha_high_res": True},
               {"fix_W_conv_high_res": True},
               {"fix_beta_high_res": False},
               {"fix_Delta_phi_high_res": False},
               {"surface_components": ["basalt"]},
               {"surface_percentage_option": "log"},
               {"thermal": False},
               {"reflection": True}]:
        out = assign_free_params(**base, **kw)
        for a, b in zip(out, out_default):
            np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# assign_free_params parity with POSEIDON
# ---------------------------------------------------------------------------
# Shared helper: build the kwargs dict for both POSEIDON's positional API
# and jaxposeidon's keyword API.
def _build_poseidon_kwargs(cfg):
    """Map jaxposeidon's keyword config to POSEIDON's positional argument order."""
    from POSEIDON.parameters import assign_free_params as poseidon_aff
    Atmosphere_dimension = max(cfg.get("PT_dim", 1), cfg.get("X_dim", 1))
    return poseidon_aff(
        cfg.get("param_species", []),
        cfg.get("bulk_species", ["H2"]),
        cfg.get("object_type", "transiting"),
        cfg["PT_profile"],
        cfg["X_profile"],
        cfg["cloud_model"],
        cfg.get("cloud_type", "deck"),
        cfg.get("gravity_setting", "fixed"),
        cfg.get("mass_setting", "fixed"),
        cfg.get("stellar_contam", None),
        cfg.get("offsets_applied", None),
        cfg.get("error_inflation", None),
        cfg.get("PT_dim", 1),
        cfg.get("X_dim", 1),
        cfg.get("cloud_dim", 1),
        cfg.get("TwoD_type", None),
        cfg.get("TwoD_param_scheme", "difference"),
        cfg.get("species_EM_gradient", []),
        cfg.get("species_DN_gradient", []),
        cfg.get("species_vert_gradient", []),
        Atmosphere_dimension,
        cfg.get("opaque_Iceberg", False),
        cfg.get("surface", False),
        cfg.get("sharp_DN_transition", False),
        cfg.get("sharp_EM_transition", False),
        cfg.get("reference_parameter", "R_p_ref"),
        cfg.get("disable_atmosphere", False),
        cfg.get("aerosol_species", []),
        cfg.get("log_P_slope_arr", [-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0]),
        cfg.get("number_P_knots", 0),
        cfg.get("PT_penalty", False),
        cfg.get("high_res_method", None),
        cfg.get("alpha_high_res_option", "log"),
        cfg.get("fix_alpha_high_res", False),
        cfg.get("fix_W_conv_high_res", False),
        cfg.get("fix_beta_high_res", True),
        cfg.get("fix_Delta_phi_high_res", True),
        cfg.get("lognormal_logwidth_free", False),
        cfg.get("surface_components", []),
        cfg.get("surface_model", "gray"),
        cfg.get("surface_percentage_option", "linear"),
        cfg.get("thermal", True),
        cfg.get("reflection", False),
    )


def _assert_assign_params_match(cfg):
    """Run both implementations and assert all 10 outputs match exactly."""
    ours = assign_free_params(**cfg)
    theirs = _build_poseidon_kwargs(cfg)
    assert len(ours) == len(theirs) == 10
    for i, (a, b) in enumerate(zip(ours, theirs)):
        np.testing.assert_array_equal(a, b,
            err_msg=f"assign_free_params output {i} differs")


def test_assign_free_params_canonical_rayleigh():
    """Pure-H2 isothermal cloud-free oracle config (POSEIDON test_TRIDENT setup)."""
    _assert_assign_params_match(dict(
        param_species=[],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    ))


def test_assign_free_params_madhu_isochem_cloud_free():
    """6-param MS09 PT, isochem, no clouds, single H2O species, R_p_ref."""
    _assert_assign_params_match(dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    ))


def test_assign_free_params_macmad17_deck_haze_1d():
    """MacMad17 deck+haze, cloud_dim=1 (full coverage, no phi_cloud)."""
    _assert_assign_params_match(dict(
        param_species=["H2O", "CH4"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="deck_haze",
        cloud_dim=1,
    ))


def test_assign_free_params_macmad17_deck_haze_cloud_dim2():
    """MacMad17 deck+haze with phi_cloud partial coverage (cloud_dim=2)."""
    _assert_assign_params_match(dict(
        param_species=["H2O", "CH4", "CO2"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="deck_haze",
        cloud_dim=2,
    ))


def test_assign_free_params_k2_18b_one_offset():
    """K2-18 b one-offset retrieval: P_ref, Madhu, MacMad17 patchy, 9 species,
    single_dataset offset."""
    _assert_assign_params_match(dict(
        param_species=["H2O", "CH4", "CO2", "CO", "NH3",
                       "HCN", "OCS", "N2O", "CH3Cl"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="deck_haze",
        cloud_dim=2,
        reference_parameter="P_ref",
        offsets_applied="single_dataset",
    ))


def test_assign_free_params_p_ref_and_rp_ref():
    """reference_parameter='R_p_ref+P_ref' yields both names."""
    _assert_assign_params_match(dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
        reference_parameter="R_p_ref+P_ref",
    ))


def test_assign_free_params_two_dataset_offset():
    """Paper has only one offset, but two_datasets is also v0-supported."""
    _assert_assign_params_match(dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
        offsets_applied="two_datasets",
    ))


def test_assign_free_params_error_inflation_combinations():
    """Line15, Piette20, and combined error-inflation variants."""
    for ei in ("Line15", "Piette20", "Line15+Piette20"):
        _assert_assign_params_match(dict(
            param_species=["H2O"],
            bulk_species=["H2"],
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="cloud-free",
            cloud_dim=1,
            error_inflation=ei,
        ))


def test_assign_free_params_haze_only():
    """MacMad17 haze-only (no deck)."""
    _assert_assign_params_match(dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="haze",
        cloud_dim=1,
    ))


def test_assign_free_params_deck_only():
    """MacMad17 deck-only (no haze)."""
    _assert_assign_params_match(dict(
        param_species=["H2O"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="deck",
        cloud_dim=1,
    ))


def test_assign_free_params_returns_numpy_arrays():
    """All returned containers are numpy arrays, matching POSEIDON's recast."""
    out = assign_free_params(
        param_species=["H2O"],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    )
    for arr in out[:-1]:
        assert isinstance(arr, np.ndarray), f"non-array output: {type(arr)}"


def test_assign_free_params_N_params_cumulative_shape():
    """N_params_cumulative has exactly 10 entries (one per group)."""
    out = assign_free_params(
        param_species=["H2O"],
        bulk_species=["H2"],
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="cloud-free",
        cloud_dim=1,
    )
    assert out[-1].shape == (10,)
    # Cumulative array is non-decreasing.
    assert np.all(np.diff(out[-1]) >= 0)


# Cloud-case tuples (cloud_model, cloud_type, cloud_dim) that the v0
# envelope accepts. cloud-free ignores cloud_type but accepts cloud_dim
# 1 or 2; MacMad17 supports {deck, haze, deck_haze} × {1, 2}.
_CLOUD_CASES = [
    ("cloud-free", "deck", 1),
    ("cloud-free", "deck", 2),
    ("MacMad17", "deck", 1),
    ("MacMad17", "deck", 2),
    ("MacMad17", "haze", 1),
    ("MacMad17", "haze", 2),
    ("MacMad17", "deck_haze", 1),
    ("MacMad17", "deck_haze", 2),
]


@pytest.mark.parametrize("PT_profile", ["isotherm", "Madhu"])
@pytest.mark.parametrize("reference_parameter",
                         ["R_p_ref", "P_ref", "R_p_ref+P_ref"])
@pytest.mark.parametrize("offsets_applied",
                         [None, "single_dataset", "two_datasets", "three_datasets"])
@pytest.mark.parametrize("error_inflation",
                         [None, "Line15", "Piette20", "Line15+Piette20"])
@pytest.mark.parametrize("cloud_case", _CLOUD_CASES)
def test_assign_free_params_full_v0_grid(PT_profile, reference_parameter,
                                          offsets_applied, error_inflation,
                                          cloud_case):
    """Combinatorial parity over all v0-supported knob combinations.

    8 cloud cases × 2 PT × 3 ref × 4 offsets × 4 err_inflation = 768
    parametric checks against POSEIDON's reference assign_free_params.
    """
    cloud_model, cloud_type, cloud_dim = cloud_case
    _assert_assign_params_match(dict(
        param_species=["H2O", "CH4"],
        bulk_species=["H2", "He"],
        PT_profile=PT_profile,
        X_profile="isochem",
        cloud_model=cloud_model,
        cloud_type=cloud_type,
        cloud_dim=cloud_dim,
        reference_parameter=reference_parameter,
        offsets_applied=offsets_applied,
        error_inflation=error_inflation,
    ))


def test_assign_free_params_split_roundtrip_with_canonical_oracle():
    """End-to-end: assign → split → POSEIDON's assign+split agree."""
    from POSEIDON.parameters import split_params as poseidon_split

    cfg = dict(
        param_species=["H2O", "CH4", "CO2"],
        bulk_species=["H2", "He"],
        PT_profile="Madhu",
        X_profile="isochem",
        cloud_model="MacMad17",
        cloud_type="deck_haze",
        cloud_dim=2,
        reference_parameter="P_ref",
        offsets_applied="single_dataset",
    )
    (_params, _phys, _PT, _X, _cl, _geo, _stel, _hr, _surf,
     N_params_cum) = assign_free_params(**cfg)
    rng = np.random.default_rng(42)
    theta = rng.standard_normal(N_params_cum[-1]).astype(np.float64)
    ours_split = split_params(theta, N_params_cum)
    theirs_split = poseidon_split(theta, N_params_cum)
    for a, b in zip(ours_split, theirs_split):
        np.testing.assert_array_equal(a, b)
