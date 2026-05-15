"""Phase 0.5.12b: Mie cloud_params + Mie_cloud runtime parity tests.

Aerosol-database (non-free/non-file_read) path, cloud_dim=1, no
shiny/uniaxial/biaxial. Compares jaxposeidon outputs against POSEIDON
`parameters.unpack_cloud_params` (Mie branch), `clouds.Mie_cloud`, and
the parameter-name ordering produced by
`POSEIDON.parameters.assign_free_params`.
"""

import os

import h5py
import numpy as np
import pytest

from jaxposeidon._clouds import (
    Mie_cloud,
    unpack_Mie_cloud_params,
)
from jaxposeidon._parameter_setup import assign_free_params


# ---------------------------------------------------------------------------
# Synthetic aerosol-grid fixture
# ---------------------------------------------------------------------------
def _synthetic_aerosol_db(
    tmp_path,
    aerosol_species=("H2O", "ZnS"),
    wl_grid=None,
    r_m_grid=None,
    seed=0,
):
    if wl_grid is None:
        wl_grid = np.linspace(0.5, 10.0, 30)
    if r_m_grid is None:
        r_m_grid = np.array([0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
    rng = np.random.default_rng(seed)
    opac_dir = tmp_path / "opacity"
    opac_dir.mkdir(exist_ok=True)
    db_path = opac_dir / "aerosol_database.hdf5"
    with h5py.File(db_path, "w") as f:
        info = f.create_group("Info")
        info.create_dataset("Wavelength grid", data=wl_grid)
        info.create_dataset("Particle Size grid", data=r_m_grid)
        for species in aerosol_species:
            sp = f.create_group(species)
            sigma_group = sp.create_group("0.5")
            for key in ("eff_ext", "eff_g", "eff_w"):
                arr = rng.uniform(0.0, 1.0, size=len(r_m_grid) * len(wl_grid))
                sigma_group.create_dataset(key, data=arr)
    return db_path, wl_grid, r_m_grid


def _atm_arrays(N_layers=30):
    P = np.logspace(2, -6, N_layers)  # 100 bar → 1e-6 bar (top of atmosphere)
    r = np.linspace(7.0e7, 8.0e7, N_layers).reshape(N_layers, 1, 1)
    H = np.full((N_layers, 1, 1), 5.0e5)
    n = np.full((N_layers, 1, 1), 1.0e20)
    return P, r, H, n


# ---------------------------------------------------------------------------
# assign_free_params: parameter-name ordering parity for the Mie block
# ---------------------------------------------------------------------------
_POSEIDON_MIE_CASES = [
    ("uniform_X", ["H2O", "ZnS"]),
    ("slab", ["H2O", "ZnS"]),
    ("fuzzy_deck", ["H2O"]),
    ("fuzzy_deck_plus_slab", ["H2O", "ZnS"]),
    ("opaque_deck_plus_slab", ["H2O", "ZnS"]),
    ("opaque_deck_plus_uniform_X", ["H2O", "ZnS"]),
    ("one_slab", ["H2O", "ZnS"]),
]


def _p_assign(param_species, cloud_model, cloud_type, cloud_dim, aerosol_species):
    """POSEIDON assign_free_params with its 44 positional defaults."""
    from POSEIDON.parameters import assign_free_params as p_assign

    return p_assign(
        param_species,
        ["H2"],  # bulk_species
        "transiting",  # object_type
        "isotherm",  # PT_profile
        "isochem",  # X_profile
        cloud_model,
        cloud_type,
        "fixed",  # gravity_setting
        "fixed",  # mass_setting
        None,  # stellar_contam
        None,  # offsets_applied
        None,  # error_inflation
        1,  # PT_dim
        1,  # X_dim
        cloud_dim,
        None,  # TwoD_type
        "difference",  # TwoD_param_scheme
        [],  # species_EM_gradient
        [],  # species_DN_gradient
        [],  # species_vert_gradient
        1,  # Atmosphere_dimension
        False,  # opaque_Iceberg
        False,  # surface
        False,  # sharp_DN_transition
        False,  # sharp_EM_transition
        "R_p_ref",  # reference_parameter
        False,  # disable_atmosphere
        aerosol_species,
        (-3.0, -2.0, -1.0, 0.0, 1.0, 1.5, 2.0),  # log_P_slope_arr
        0,  # number_P_knots
        False,  # PT_penalty
        None,  # high_res_method
        "log",  # alpha_high_res_option
        False,  # fix_alpha_high_res
        False,  # fix_W_conv_high_res
        True,  # fix_beta_high_res
        True,  # fix_Delta_phi_high_res
        False,  # lognormal_logwidth_free
        [],  # surface_components
        "gray",  # surface_model
        "linear",  # surface_percentage_option
        True,  # thermal
        False,  # reflection
    )


@pytest.mark.parametrize("cloud_type,aerosol_species", _POSEIDON_MIE_CASES)
def test_assign_free_params_Mie_cloud_block_matches_poseidon(
    cloud_type, aerosol_species
):
    param_species = ["H2O"]
    ours = assign_free_params(
        param_species=param_species,
        PT_profile="isotherm",
        X_profile="isochem",
        cloud_model="Mie",
        cloud_type=cloud_type,
        cloud_dim=1,
        aerosol_species=aerosol_species,
    )
    theirs = _p_assign(param_species, "Mie", cloud_type, 1, aerosol_species)
    # POSEIDON returns the cloud_params array at index 4; we match.
    np.testing.assert_array_equal(np.asarray(ours[4]), np.asarray(theirs[4]))


# ---------------------------------------------------------------------------
# unpack_Mie_cloud_params parity vs POSEIDON unpack_cloud_params
# ---------------------------------------------------------------------------
def _poseidon_mie_unpack(*, cloud_param_names, clouds_in, cloud_dim=1):
    from POSEIDON.parameters import unpack_cloud_params as p_unpack

    n_cp = len(cloud_param_names)
    N_params_cumulative = np.array([0, 0, 0, n_cp, n_cp, n_cp, n_cp, n_cp, n_cp, n_cp])
    param_names = np.array(list(cloud_param_names))
    return p_unpack(
        param_names,
        np.asarray(clouds_in, dtype=float),
        "Mie",
        cloud_dim,
        N_params_cumulative,
        None,
    )


def test_unpack_Mie_cloud_params_uniform_X_matches_poseidon():
    aerosols = ["H2O", "ZnS"]
    names = [
        "log_r_m_H2O",
        "log_X_H2O",
        "log_r_m_ZnS",
        "log_X_ZnS",
    ]
    vals = [-1.0, -4.0, -0.5, -6.0]
    ours = unpack_Mie_cloud_params(
        clouds_in=vals,
        cloud_param_names=np.array(names),
        cloud_type="uniform_X",
        cloud_dim=1,
        aerosol_species=aerosols,
    )
    theirs = _poseidon_mie_unpack(cloud_param_names=names, clouds_in=vals)
    # POSEIDON return slot indices (parameters.py:2475-2478):
    # 0 kappa_cloud_0, 1 P_cloud, 7 r_m, 12 log_X_Mie
    np.testing.assert_allclose(ours["r_m"], np.asarray(theirs[7]))
    np.testing.assert_allclose(ours["log_X_Mie"], np.asarray(theirs[12]))
    assert ours["kappa_cloud_0"] == theirs[0]


def test_unpack_Mie_cloud_params_slab_matches_poseidon():
    aerosols = ["H2O"]
    names = [
        "log_P_top_slab_H2O",
        "Delta_log_P_H2O",
        "log_r_m_H2O",
        "log_X_H2O",
    ]
    vals = [-2.0, 1.5, -1.0, -4.0]
    ours = unpack_Mie_cloud_params(
        clouds_in=vals,
        cloud_param_names=np.array(names),
        cloud_type="slab",
        cloud_dim=1,
        aerosol_species=aerosols,
    )
    theirs = _poseidon_mie_unpack(cloud_param_names=names, clouds_in=vals)
    # P_cloud, P_slab_bottom (slot 13), r_m, log_X_Mie
    np.testing.assert_allclose(ours["P_cloud"], np.asarray(theirs[1]))
    np.testing.assert_allclose(ours["P_slab_bottom"], np.asarray(theirs[13]))
    np.testing.assert_allclose(ours["r_m"], np.asarray(theirs[7]))
    np.testing.assert_allclose(ours["log_X_Mie"], np.asarray(theirs[12]))


def test_unpack_Mie_cloud_params_fuzzy_deck_matches_poseidon():
    aerosols = ["H2O"]
    names = [
        "log_P_top_deck_H2O",
        "log_r_m_H2O",
        "log_n_max_H2O",
        "f_H2O",
    ]
    vals = [-3.0, -1.0, 5.0, 0.5]
    ours = unpack_Mie_cloud_params(
        clouds_in=vals,
        cloud_param_names=np.array(names),
        cloud_type="fuzzy_deck",
        cloud_dim=1,
        aerosol_species=aerosols,
    )
    theirs = _poseidon_mie_unpack(cloud_param_names=names, clouds_in=vals)
    np.testing.assert_allclose(ours["P_cloud"], np.asarray(theirs[1]))
    np.testing.assert_allclose(ours["r_m"], np.asarray(theirs[7]))
    np.testing.assert_allclose(ours["log_n_max"], np.asarray(theirs[8]))
    np.testing.assert_allclose(ours["fractional_scale_height"], np.asarray(theirs[9]))


# ---------------------------------------------------------------------------
# Mie_cloud runtime parity vs POSEIDON Mie_cloud
# ---------------------------------------------------------------------------
def _aerosol_grid(tmp_path, monkeypatch, species=("H2O", "ZnS")):
    from jaxposeidon._aerosol_db_loader import load_aerosol_grid

    _synthetic_aerosol_db(tmp_path, aerosol_species=species)
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path) + os.sep)
    return load_aerosol_grid(list(species))


@pytest.mark.parametrize(
    "cloud_type,n_species",
    [("uniform_X", 2), ("slab", 2), ("fuzzy_deck", 1), ("one_slab", 2)],
)
def test_Mie_cloud_runtime_matches_poseidon(
    tmp_path, monkeypatch, cloud_type, n_species
):
    from POSEIDON.clouds import Mie_cloud as p_Mie

    species = ["H2O", "ZnS"][:n_species]
    grid = _aerosol_grid(tmp_path, monkeypatch, species=tuple(species))
    P, r, H, n = _atm_arrays()
    wl = np.linspace(1.0, 5.0, 25)
    r_m = np.array([0.1] * n_species)
    log_X_Mie = np.array([-4.0] * n_species)
    P_cloud = np.array([1e-2] * n_species)
    P_cloud_bottom = np.array([1.0] * n_species)
    log_n_max = np.array([5.0] * n_species)
    fractional_scale_height = np.array([0.5] * n_species)

    common = dict(
        P=P,
        wl=wl,
        r=r,
        H=H,
        n=n,
        r_m=r_m,
        aerosol_species=species,
        cloud_type=cloud_type,
        aerosol_grid=grid,
    )
    if cloud_type == "fuzzy_deck":
        kwargs = dict(
            P_cloud=P_cloud,
            log_n_max=log_n_max,
            fractional_scale_height=fractional_scale_height,
        )
    elif cloud_type == "uniform_X":
        kwargs = dict(log_X_Mie=log_X_Mie)
    elif cloud_type == "slab":
        kwargs = dict(
            log_X_Mie=log_X_Mie,
            P_cloud=P_cloud,
            P_cloud_bottom=P_cloud_bottom,
        )
    elif cloud_type == "one_slab":
        # one_slab: scalar P_cloud / P_cloud_bottom
        kwargs = dict(
            log_X_Mie=log_X_Mie,
            P_cloud=1e-2,
            P_cloud_bottom=1.0,
        )

    n_aer_ours, sigma_ours, g_ours, w_ours = Mie_cloud(**common, **kwargs)
    n_aer_theirs, sigma_theirs, g_theirs, w_theirs = p_Mie(**common, **kwargs)

    for aer in range(len(n_aer_ours)):
        np.testing.assert_allclose(n_aer_ours[aer], n_aer_theirs[aer], rtol=0, atol=0)
    for aer in range(len(sigma_ours)):
        np.testing.assert_allclose(
            sigma_ours[aer], sigma_theirs[aer], rtol=1e-12, atol=0
        )
        np.testing.assert_allclose(g_ours[aer], g_theirs[aer], rtol=1e-12)
        np.testing.assert_allclose(w_ours[aer], w_theirs[aer], rtol=1e-12)


# ---------------------------------------------------------------------------
# Guards: deferred branches still raise NotImplementedError
# ---------------------------------------------------------------------------
def test_unpack_Mie_cloud_params_rejects_deferred_cloud_types():
    for cloud_type in ("uniaxial_slab", "biaxial_random_slab", "shiny_fuzzy_deck"):
        with pytest.raises(NotImplementedError, match="Mie"):
            unpack_Mie_cloud_params(
                clouds_in=[0.0],
                cloud_param_names=np.array(["foo"]),
                cloud_type=cloud_type,
                cloud_dim=1,
                aerosol_species=["H2O"],
            )


def test_unpack_Mie_cloud_params_rejects_patchy_cloud_dim():
    with pytest.raises(NotImplementedError, match="cloud_dim"):
        unpack_Mie_cloud_params(
            clouds_in=[-1.0, -4.0],
            cloud_param_names=np.array(["log_r_m_H2O", "log_X_H2O"]),
            cloud_type="uniform_X",
            cloud_dim=2,
            aerosol_species=["H2O"],
        )


def test_unpack_Mie_cloud_params_rejects_free_aerosol_species():
    with pytest.raises(NotImplementedError, match="free"):
        unpack_Mie_cloud_params(
            clouds_in=[-1.0],
            cloud_param_names=np.array(["log_r_m"]),
            cloud_type="uniform_X",
            cloud_dim=1,
            aerosol_species=["free"],
        )


def test_assign_free_params_Mie_rejects_unsupported_cloud_type():
    with pytest.raises(NotImplementedError, match="Mie cloud_type"):
        assign_free_params(
            param_species=["H2O"],
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="Mie",
            cloud_type="biaxial_slab",
            cloud_dim=1,
            aerosol_species=["H2O", "ZnS"],
        )


def test_assign_free_params_Mie_requires_aerosol_species():
    with pytest.raises(Exception, match="non-empty aerosol_species"):
        assign_free_params(
            param_species=["H2O"],
            PT_profile="isotherm",
            X_profile="isochem",
            cloud_model="Mie",
            cloud_type="uniform_X",
            cloud_dim=1,
            aerosol_species=[],
        )
