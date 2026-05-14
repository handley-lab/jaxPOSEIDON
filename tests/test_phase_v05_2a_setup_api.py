"""Phase 0.5.2a: POSEIDON setup API parity tests.

Native ports in `_setup_api.py`:
  - `wl_grid_constant_R` (pure arithmetic)
  - `create_star` (blackbody path; non-blackbody → 0.5.11)
  - `create_planet` (fixed gravity/mass; free → 0.5.2b)

Thin POSEIDON delegators (real ports follow in 0.5.6 / 0.5.7 / 0.5.8 /
0.5.9 / 0.5.11 / 0.5.12 / 0.5.14 / 0.5.15):
  - `define_model`
  - `read_opacities`
  - `make_atmosphere`

Tests use POSEIDON as the parity oracle on a v0-compatible config.
"""

import numpy as np
import pytest

import jaxposeidon as jpo


# ---------------------------------------------------------------------------
# Native ports — direct POSEIDON parity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "wl_min,wl_max,R",
    [
        (0.5, 5.0, 1000),
        (1.0, 3.0, 100),
        (0.2, 10.0, 20000),
        (0.8, 4.5, 5000),
    ],
)
def test_wl_grid_constant_R_matches_poseidon(wl_min, wl_max, R):
    from POSEIDON.core import wl_grid_constant_R as p_wl

    ours = jpo.wl_grid_constant_R(wl_min, wl_max, R)
    theirs = p_wl(wl_min, wl_max, R)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize(
    "R_s,T_eff,log_g,Met",
    [
        (6.96e8, 5772.0, 4.44, 0.0),    # Sun
        (5.96e8, 4500.0, 4.6, -0.1),   # K-dwarf
        (1.0e9, 6500.0, 4.0, 0.3),     # F-type
    ],
)
def test_create_star_blackbody_matches_poseidon(R_s, T_eff, log_g, Met):
    from POSEIDON.core import create_star as p_create_star

    ours = jpo.create_star(R_s, T_eff, log_g, Met)
    theirs = p_create_star(R_s, T_eff, log_g, Met)

    # Compare scalar / vector fields
    for k in ("R_s", "T_eff", "T_eff_error", "log_g_error", "Met", "log_g",
              "stellar_grid", "stellar_contam"):
        assert ours[k] == theirs[k], k
    np.testing.assert_array_equal(ours["wl_star"], theirs["wl_star"])
    np.testing.assert_allclose(ours["I_phot"], theirs["I_phot"],
                                atol=0, rtol=1e-14)
    np.testing.assert_allclose(ours["F_star"], theirs["F_star"],
                                atol=0, rtol=1e-14)


def test_create_star_stellar_contam_rejected():
    with pytest.raises(NotImplementedError, match="Stellar contamination"):
        jpo.create_star(7e8, 5500, 4.4, 0.0, stellar_contam="one_spot")


def test_create_star_non_blackbody_rejected():
    with pytest.raises(NotImplementedError, match="stellar_grid"):
        jpo.create_star(7e8, 5500, 4.4, 0.0, stellar_grid="phoenix")


# ---------------------------------------------------------------------------
# create_planet — fixed gravity/mass path
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mass=1.898e27),                          # mass only
        dict(gravity=24.79),                          # gravity only
        dict(log_g=3.394),                            # log_g only
        dict(mass=1.898e27, b_p=0.1, a_p=7.78e11),    # full
        dict(mass=5.972e24, T_eq=288.0, d=1.496e11),  # Earth-like
    ],
)
def test_create_planet_matches_poseidon(kwargs):
    from POSEIDON.core import create_planet as p_create_planet

    R_p = 6.4e7
    ours = jpo.create_planet("test", R_p, **kwargs)
    theirs = p_create_planet("test", R_p, **kwargs)

    for k in ("planet_name", "planet_radius", "planet_T_eq",
              "planet_impact_parameter", "system_distance",
              "system_distance_error", "planet_semi_major_axis"):
        assert ours[k] == theirs[k] or (
            ours[k] is None and theirs[k] is None
        ), k
    np.testing.assert_allclose(ours["planet_radius"], theirs["planet_radius"],
                                atol=0, rtol=0)
    np.testing.assert_allclose(ours["planet_mass"], theirs["planet_mass"],
                                atol=0, rtol=1e-14)
    np.testing.assert_allclose(ours["planet_gravity"], theirs["planet_gravity"],
                                atol=0, rtol=1e-14)


def test_create_planet_requires_mass_or_gravity():
    with pytest.raises(Exception, match="Mass or gravity"):
        jpo.create_planet("test", 6.4e7)


# ---------------------------------------------------------------------------
# Thin POSEIDON delegators (real ports in subsequent phases)
# ---------------------------------------------------------------------------
def test_define_model_delegates_to_poseidon():
    """v0.5.2a thin delegator. 0.5.6+ replace this with a native impl."""
    m_jpo = jpo.define_model("m", ["H2"], [], PT_profile="isotherm")
    from POSEIDON.core import define_model as p_define
    m_pos = p_define("m", ["H2"], [], PT_profile="isotherm")
    # Same keys and types
    assert set(m_jpo.keys()) == set(m_pos.keys())


def test_read_opacities_delegates_to_poseidon(tmp_path):
    """v0.5.2a thin delegator. 0.5.4/0.5.12/0.5.15 replace this."""
    import os
    import h5py

    opac_dir = tmp_path / "opacity"
    opac_dir.mkdir()
    cia_path = opac_dir / "Opacity_database_cia.hdf5"
    T_grid = np.linspace(200, 2000, 10, dtype=np.float64)
    nu = np.linspace(1.0e4, 5.0e5, 50, dtype=np.float64)
    log_cia = np.full((len(T_grid), len(nu)), -50.0, dtype=np.float64)
    with h5py.File(cia_path, "w") as f:
        for pair in ("H2-H2", "H2-He"):
            g = f.create_group(pair)
            g.create_dataset("T", data=T_grid)
            g.create_dataset("nu", data=nu)
            g.create_dataset("log(cia)", data=log_cia)

    saved = os.environ.get("POSEIDON_input_data")
    os.environ["POSEIDON_input_data"] = str(tmp_path)
    try:
        m = jpo.define_model("m", ["H2", "He"], [], PT_profile="isotherm")
        wl = jpo.wl_grid_constant_R(1.0, 3.0, 500)
        opac = jpo.read_opacities(
            m, wl, "opacity_sampling",
            np.arange(700, 1110, 20), np.arange(-6.0, 2.2, 0.4),
            testing=True,
        )
        assert "sigma_stored" in opac
    finally:
        if saved is None:
            del os.environ["POSEIDON_input_data"]
        else:
            os.environ["POSEIDON_input_data"] = saved


def test_make_atmosphere_delegates_to_poseidon():
    """v0.5.2a thin delegator. 0.5.6/0.5.7/0.5.8/0.5.9 replace this."""
    planet = jpo.create_planet("p", 6.4e7, mass=1.898e27)
    m = jpo.define_model("m", ["H2"], [], PT_profile="isotherm")
    P = np.logspace(2, -7, 60)
    atm = jpo.make_atmosphere(
        planet, m, P, 10.0, 6.4e7,
        np.array([900.0]), np.array([]),
        constant_gravity=True,
    )
    assert "P" in atm
    assert "T" in atm
    np.testing.assert_array_equal(atm["P"], P)
