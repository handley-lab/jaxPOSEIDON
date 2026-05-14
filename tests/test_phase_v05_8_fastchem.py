"""Phase 0.5.8: FastChem equilibrium-chemistry grid loader + interpolation.

Ports POSEIDON `chemistry.py:16-271`:
- `_fastchem_grid_loader.load_chemistry_grid` (file I/O setup-only)
- `_chemistry.interpolate_log_X_grid` (RegularGridInterpolator runtime)

Synthetic HDF5 fixtures are built in `tmp_path`. The real ~1 GB
FastChem grid lives at `$POSEIDON_input_data/chemistry_grids/
fastchem_database.hdf5`; the env-gated `poseidon_input_data` fixture
in conftest.py provides it where available.
"""

import os

import h5py
import numpy as np
import pytest

from jaxposeidon import _chemistry, _fastchem_grid_loader


def _synthetic_fastchem_grid(
    tmp_path,
    chemical_species=("H2O", "CH4"),
    T_grid=np.array([300.0, 1000.0, 2000.0, 4000.0]),
    P_grid=np.array([1e-6, 1e-3, 1.0, 100.0]),
    Met_grid=np.array([0.1, 1.0, 10.0]),
    C_to_O_grid=np.array([0.3, 0.55, 1.0]),
):
    """Write a synthetic chemistry grid HDF5 in the POSEIDON layout."""
    rng = np.random.default_rng(0)
    chem_dir = tmp_path / "chemistry_grids"
    chem_dir.mkdir()
    db_path = chem_dir / "fastchem_database.hdf5"
    with h5py.File(db_path, "w") as f:
        info = f.create_group("Info")
        info.create_dataset("T grid", data=T_grid)
        info.create_dataset("P grid", data=P_grid)
        info.create_dataset("M/H grid", data=Met_grid)
        info.create_dataset("C/O grid", data=C_to_O_grid)
        for species in chemical_species:
            sp = f.create_group(species)
            arr = rng.uniform(
                -8.0,
                -1.0,
                size=(len(Met_grid) * len(C_to_O_grid) * len(T_grid) * len(P_grid)),
            )
            sp.create_dataset("log(X)", data=arr)
    return db_path


def test_load_chemistry_grid_synthetic_fixture(tmp_path, monkeypatch):
    _synthetic_fastchem_grid(tmp_path)
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    grid = _fastchem_grid_loader.load_chemistry_grid(["H2O", "CH4"])
    assert grid["grid"] == "fastchem"
    assert grid["log_X_grid"].shape == (2, 3, 3, 4, 4)  # (N_sp, Met, C/O, T, P)
    np.testing.assert_array_equal(
        grid["T_grid"], np.array([300.0, 1000.0, 2000.0, 4000.0])
    )


def test_load_chemistry_grid_rejects_unknown_grid():
    with pytest.raises(Exception, match="unsupported chemistry grid"):
        _fastchem_grid_loader.load_chemistry_grid(["H2O"], grid="unknown")


def test_load_chemistry_grid_rejects_unsupported_species(tmp_path, monkeypatch):
    _synthetic_fastchem_grid(tmp_path)
    monkeypatch.setenv("POSEIDON_input_data", str(tmp_path))
    with pytest.raises(Exception, match="not supported"):
        _fastchem_grid_loader.load_chemistry_grid(["NotARealSpecies"])


def test_load_chemistry_grid_requires_env_var(monkeypatch):
    monkeypatch.delenv("POSEIDON_input_data", raising=False)
    with pytest.raises(Exception, match="POSEIDON_input_data"):
        _fastchem_grid_loader.load_chemistry_grid(["H2O"])


# ---------------------------------------------------------------------------
# Interpolation parity vs POSEIDON
# ---------------------------------------------------------------------------
def _build_grid_dict(rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    T_grid = np.array([300.0, 1000.0, 2000.0, 4000.0])
    P_grid = np.array([1e-6, 1e-3, 1.0, 100.0])
    Met_grid = np.array([0.1, 1.0, 10.0])
    C_to_O_grid = np.array([0.3, 0.55, 1.0])
    log_X_grid = rng.uniform(-8.0, -1.0, size=(2, 3, 3, 4, 4))
    return {
        "grid": "fastchem",
        "log_X_grid": log_X_grid,
        "T_grid": T_grid,
        "P_grid": P_grid,
        "Met_grid": Met_grid,
        "C_to_O_grid": C_to_O_grid,
    }


def test_interpolate_log_X_grid_scalar_input_matches_poseidon():
    from POSEIDON.chemistry import interpolate_log_X_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "CH4"])
    args = dict(
        chemistry_grid=grid,
        log_P=np.array([0.0]),
        T=np.array([1500.0]),
        C_to_O=0.55,
        log_Met=0.0,
        chemical_species=species,
        return_dict=False,
    )
    ours = _chemistry.interpolate_log_X_grid(**args)
    theirs = p_interp(**args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_interpolate_log_X_grid_array_input_matches_poseidon():
    from POSEIDON.chemistry import interpolate_log_X_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "CH4"])
    log_P = np.linspace(-5.0, 1.0, 30)
    T = np.linspace(500.0, 3000.0, 30)
    args = dict(
        chemistry_grid=grid,
        log_P=log_P,
        T=T,
        C_to_O=0.55,
        log_Met=0.0,
        chemical_species=species,
        return_dict=False,
    )
    ours = _chemistry.interpolate_log_X_grid(**args)
    theirs = p_interp(**args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_interpolate_log_X_grid_3D_T_matches_poseidon():
    """POSEIDON's standard 3D temperature field with (N_layers, N_sectors, N_zones)."""
    from POSEIDON.chemistry import interpolate_log_X_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "CH4"])
    N_layers, N_sectors, N_zones = 20, 1, 1
    log_P = np.linspace(-5.0, 1.0, N_layers)
    T = np.linspace(500.0, 3000.0, N_layers).reshape(N_layers, N_sectors, N_zones)
    args = dict(
        chemistry_grid=grid,
        log_P=log_P,
        T=T,
        C_to_O=0.55,
        log_Met=0.0,
        chemical_species=species,
        return_dict=False,
    )
    ours = _chemistry.interpolate_log_X_grid(**args)
    theirs = p_interp(**args)
    np.testing.assert_allclose(ours, theirs, atol=0, rtol=1e-13)


def test_interpolate_log_X_grid_return_dict_matches_poseidon():
    from POSEIDON.chemistry import interpolate_log_X_grid as p_interp

    grid = _build_grid_dict()
    species = np.array(["H2O", "CH4"])
    args = dict(
        chemistry_grid=grid,
        log_P=np.array([0.0, -2.0]),
        T=np.array([1500.0, 1000.0]),
        C_to_O=0.55,
        log_Met=0.0,
        chemical_species=species,
        return_dict=True,
    )
    ours = _chemistry.interpolate_log_X_grid(**args)
    theirs = p_interp(**args)
    assert set(ours.keys()) == set(theirs.keys())
    for k in ours:
        np.testing.assert_allclose(ours[k], theirs[k], atol=0, rtol=1e-13)


def test_interpolate_log_X_grid_rejects_out_of_bounds():
    grid = _build_grid_dict()
    species = np.array(["H2O"])
    with pytest.raises(Exception, match="pressure"):
        _chemistry.interpolate_log_X_grid(
            grid,
            log_P=np.array([5.0]),
            T=1500.0,
            C_to_O=0.55,
            log_Met=0.0,
            chemical_species=species,
            return_dict=False,
        )
    with pytest.raises(Exception, match="temperature"):
        _chemistry.interpolate_log_X_grid(
            grid,
            log_P=0.0,
            T=10000.0,
            C_to_O=0.55,
            log_Met=0.0,
            chemical_species=species,
            return_dict=False,
        )


@pytest.mark.skipif(
    os.environ.get("POSEIDON_input_data") is None
    or not os.path.exists(
        os.path.join(
            os.environ.get("POSEIDON_input_data", ""),
            "chemistry_grids",
            "fastchem_database.hdf5",
        )
    ),
    reason="Real FastChem grid not available; set POSEIDON_input_data to enable.",
)
def test_real_fastchem_grid_loads_and_interpolates():
    """Env-gated smoke test against the real ~1 GB FastChem grid."""
    grid = _fastchem_grid_loader.load_chemistry_grid(["H2O", "CO", "CH4", "CO2"])
    log_X = _chemistry.interpolate_log_X_grid(
        grid,
        log_P=np.array([0.0]),
        T=np.array([1500.0]),
        C_to_O=0.55,
        log_Met=0.0,
        chemical_species=np.array(["H2O", "CO", "CH4", "CO2"]),
        return_dict=False,
    )
    assert log_X.shape[0] == 4
    assert np.all(np.isfinite(log_X))
    assert np.all(log_X < 0.0)  # All log10(X) should be below 0 for trace species
