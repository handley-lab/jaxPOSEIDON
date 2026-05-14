"""Phase 0.5.12 (subset): Mie aerosol grid loader + interpolator.

Ports POSEIDON `clouds.py:1461-1775` for the default `aerosol` grid
and fixed lognormal-width (0.5) case. The full Mie_cloud integration
with the runtime forward model lives in phase 0.5.13e.

Real aerosol DB lives at Zenodo DOI 10.5281/zenodo.15711943; env-gated
real-grid smoke test loads `$POSEIDON_input_data/opacity/aerosol_database.hdf5`.
"""

import os

import h5py
import numpy as np
import pytest

from jaxposeidon import _clouds, _aerosol_db_loader


def _synthetic_aerosol_db(
    tmp_path,
    aerosol_species=("H2O", "ZnS"),
    wl_grid=np.linspace(0.5, 10.0, 20),
    r_m_grid=np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
):
    rng = np.random.default_rng(0)
    opac_dir = tmp_path / "opacity"
    opac_dir.mkdir()
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


def test_load_aerosol_grid_synthetic_fixture(tmp_path, monkeypatch):
    _synthetic_aerosol_db(tmp_path)
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    grid = _aerosol_db_loader.load_aerosol_grid(["H2O", "ZnS"])
    assert grid["grid"] == "aerosol"
    assert grid["sigma_Mie_grid"].shape == (2, 3, 6, 20)


def test_load_aerosol_grid_matches_poseidon(tmp_path, monkeypatch):
    """POSEIDON's load_aerosol_grid expects a trailing slash on
    POSEIDON_input_data (it does string concat `input_file_path + 'opacity/'`).
    The port uses os.path.join, so set the env var with explicit trailing
    separator to make POSEIDON happy."""
    from POSEIDON.clouds import load_aerosol_grid as p_load

    _synthetic_aerosol_db(tmp_path)
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path) + os.sep)
    ours = _aerosol_db_loader.load_aerosol_grid(["H2O", "ZnS"])
    theirs = p_load(["H2O", "ZnS"])
    assert ours["grid"] == theirs["grid"]
    np.testing.assert_array_equal(ours["wl_grid"], theirs["wl_grid"])
    np.testing.assert_array_equal(ours["r_m_grid"], theirs["r_m_grid"])
    np.testing.assert_array_equal(
        ours["sigma_Mie_grid"], np.asarray(theirs["sigma_Mie_grid"])
    )


def test_load_aerosol_grid_rejects_deferred_grids():
    """Non-default grids (SiO2_free_logwidth / aerosol_directional /
    aerosol_diamonds) are POSEIDON-supported but not yet implemented here."""
    for grid in ("SiO2_free_logwidth", "aerosol_directional", "aerosol_diamonds"):
        with pytest.raises(Exception, match="unsupported aerosol grid"):
            _aerosol_db_loader.load_aerosol_grid(["H2O"], grid=grid)


def test_load_aerosol_grid_rejects_unknown_grid():
    with pytest.raises(Exception, match="unsupported aerosol grid"):
        _aerosol_db_loader.load_aerosol_grid(["H2O"], grid="unknown")


def test_load_aerosol_grid_requires_env_var(monkeypatch):
    monkeypatch.delenv("POSEIDON_input_data", raising=False)
    with pytest.raises(Exception, match="POSEIDON_input_data"):
        _aerosol_db_loader.load_aerosol_grid(["H2O"])


def test_load_aerosol_grid_missing_db_raises(tmp_path, monkeypatch):
    """No HDF5 file → descriptive error."""
    (tmp_path / "opacity").mkdir()
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    with pytest.raises(Exception, match="could not find"):
        _aerosol_db_loader.load_aerosol_grid(["H2O"])


def _build_grid_dict(rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    wl_grid = np.linspace(0.5, 10.0, 20)
    r_m_grid = np.array([0.01, 0.05, 0.1, 0.5, 1.0, 5.0])
    sigma_Mie_grid = rng.uniform(0.0, 1.0, size=(2, 3, len(r_m_grid), len(wl_grid)))
    return {
        "grid": "aerosol",
        "sigma_Mie_grid": sigma_Mie_grid,
        "wl_grid": wl_grid,
        "r_m_grid": r_m_grid,
    }


def test_interpolate_sigma_Mie_grid_array_input_matches_poseidon():
    from POSEIDON.clouds import interpolate_sigma_Mie_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "ZnS"])
    wl = np.linspace(1.0, 5.0, 30)
    r_m_array = np.array([0.1, 0.5])
    ours = _clouds.interpolate_sigma_Mie_grid(
        grid,
        wl,
        r_m_array,
        species,
        return_dict=False,
    )
    theirs = p_interp(grid, wl, r_m_array, species, return_dict=False)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_interpolate_sigma_Mie_grid_return_dict_matches_poseidon():
    from POSEIDON.clouds import interpolate_sigma_Mie_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "ZnS"])
    wl = np.linspace(1.0, 5.0, 30)
    r_m_array = np.array([0.1, 0.5])
    ours = _clouds.interpolate_sigma_Mie_grid(
        grid,
        wl,
        r_m_array,
        species,
        return_dict=True,
    )
    theirs = p_interp(grid, wl, r_m_array, species, return_dict=True)
    assert set(ours.keys()) == set(theirs.keys())
    for sp in ("H2O", "ZnS"):
        for key in ("eff_ext", "eff_g", "eff_w"):
            np.testing.assert_allclose(
                ours[sp][key], theirs[sp][key], atol=0, rtol=1e-13
            )


def test_interpolate_sigma_Mie_grid_single_string_species_array_equivalence():
    """Single string `aerosol_species` argument equivalence vs single-element
    array (POSEIDON's path crashes on string + np.array conversion)."""
    grid = _build_grid_dict()
    wl = np.linspace(1.0, 5.0, 30)
    r_m_array = np.array([0.1])
    out_string = _clouds.interpolate_sigma_Mie_grid(
        grid,
        wl,
        r_m_array,
        "H2O",
        return_dict=False,
    )
    out_array = _clouds.interpolate_sigma_Mie_grid(
        grid,
        wl,
        r_m_array,
        np.array(["H2O"]),
        return_dict=False,
    )
    # Both paths should return identical eff_ext/eff_g/eff_w stacks
    np.testing.assert_array_equal(np.asarray(out_string), out_array[0])


def test_interpolate_sigma_Mie_grid_rejects_out_of_bounds():
    grid = _build_grid_dict()
    species = np.array(["H2O"])
    # Out-of-range wavelength
    with pytest.raises(Exception, match="wavelength"):
        _clouds.interpolate_sigma_Mie_grid(
            grid,
            np.array([20.0, 30.0]),
            np.array([0.1]),
            species,
            return_dict=False,
        )
    # Out-of-range particle size
    with pytest.raises(Exception, match="particle size"):
        _clouds.interpolate_sigma_Mie_grid(
            grid,
            np.array([1.0]),
            np.array([100.0]),
            species,
            return_dict=False,
        )


@pytest.mark.skipif(
    os.environ.get("POSEIDON_input_data") is None
    or not os.path.exists(
        os.path.join(
            os.environ.get("POSEIDON_input_data", ""),
            "opacity",
            "aerosol_database.hdf5",
        )
    ),
    reason="Real aerosol DB not available; set POSEIDON_input_data to enable.",
)
def test_real_aerosol_grid_loads_and_interpolates():
    """Env-gated smoke test against the real Zenodo aerosol grid."""
    grid = _aerosol_db_loader.load_aerosol_grid(["H2O", "ZnS"])
    out = _clouds.interpolate_sigma_Mie_grid(
        grid,
        np.linspace(1.0, 5.0, 50),
        np.array([0.1, 0.5]),
        np.array(["H2O", "ZnS"]),
        return_dict=False,
    )
    assert out.shape[0] == 2  # 2 species
    assert np.all(np.isfinite(out))
