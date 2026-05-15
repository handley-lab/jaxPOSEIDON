"""Phase 0 scaffold tests — imports, oracle availability."""

import pytest


def test_jaxposeidon_imports():
    """Package and stub modules import cleanly."""
    import jaxposeidon
    from jaxposeidon import (
        _opacity_precompute,
        _opacities,
        _atmosphere,
        _chemistry,
        _clouds,
        _instruments,
        _geometry,
        _transmission,
        _parameters,
        _data,
        _priors,
        _constants,
        _species_data,
        _loaddata,
        _compute_spectrum,
        core,
    )

    assert jaxposeidon.__version__.split(".")[0].isdigit()
    for mod in (
        _opacity_precompute,
        _opacities,
        _atmosphere,
        _chemistry,
        _clouds,
        _instruments,
        _geometry,
        _transmission,
        _parameters,
        _data,
        _priors,
        _constants,
        _species_data,
        _loaddata,
        _compute_spectrum,
        core,
    ):
        assert mod is not None


def test_poseidon_oracle_available():
    """POSEIDON imports — required for all subsequent phase tests."""
    import POSEIDON
    from POSEIDON.core import (
        create_star,
        create_planet,
        define_model,
        read_opacities,
        make_atmosphere,
        compute_spectrum,
    )

    assert all(
        callable(f)
        for f in (
            create_star,
            create_planet,
            define_model,
            read_opacities,
            make_atmosphere,
            compute_spectrum,
        )
    )


K2_18B_TARGETS = {
    "H2O",
    "CH4",
    "CO2",
    "CO",
    "NH3",
    "HCN",
    "OCS",
    "N2O",
    "CH3Cl",
    "CS2",
    "C2H6S",
}


def test_k2_18b_species_supported_in_poseidon():
    """K2-18 b target species are in POSEIDON's supported_species table.

    DMS = C2H6S per POSEIDON naming convention (DMDS = C2H6S2 also present).
    """
    from POSEIDON.supported_chemicals import supported_species

    missing = K2_18B_TARGETS - set(supported_species.tolist())
    assert not missing, f"Missing K2-18b target species in POSEIDON: {missing}"


def test_k2_18b_species_in_opacity_hdf5(poseidon_input_data):
    """K2-18 b target species exist as groups in Opacity_database_v1.3.hdf5
    and each group carries the schema documented in the plan:
    `T`, `log(P)`, `nu`, `log(sigma)` (POSEIDON/POSEIDON/absorption.py:929-943).
    """
    import os
    import h5py

    db_path = os.path.join(poseidon_input_data, "opacity", "Opacity_database_v1.3.hdf5")
    if not os.path.isfile(db_path):
        import pytest

        pytest.skip(f"{db_path} not found; opacity DB not installed at this version.")

    required_datasets = ("T", "log(P)", "nu", "log(sigma)")
    with h5py.File(db_path, "r") as f:
        groups_present = set(f.keys())
        missing_groups = K2_18B_TARGETS - groups_present
        assert not missing_groups, (
            f"Missing species groups in {db_path}: {missing_groups}"
        )
        for sp in sorted(K2_18B_TARGETS):
            for ds in required_datasets:
                assert ds in f[sp], (
                    f"{sp}/{ds} missing in {db_path} (schema per absorption.py:929-943)"
                )


def test_paired_oracle_harness_shape(monkeypatch):
    """tests/oracle.paired_transmission_spectra wires both sides correctly.

    Phase 0 stubs the JAX side with a NotImplementedError-raising callable;
    the harness must propagate that without swallowing it. Each subsequent
    phase will replace the stub with progressively more port code.

    Uses monkeypatch to stub POSEIDON's side too, so this test does not
    require POSEIDON_input_data or the CIA HDF5 — it only verifies the
    paired-harness control flow.
    """
    from tests import oracle

    monkeypatch.setattr(oracle, "canonical_rayleigh_config", lambda: {"wl": [0.0]})
    monkeypatch.setattr(oracle, "poseidon_transmission_spectrum", lambda cfg: [0.0])

    def jax_stub(cfg):
        raise NotImplementedError("jaxposeidon forward model: Phase 1+")

    with pytest.raises(NotImplementedError):
        oracle.paired_transmission_spectra(jax_stub)
