"""Phase 0 scaffold tests — imports, oracle availability."""

def test_jaxposeidon_imports():
    """Package and stub modules import cleanly."""
    import jaxposeidon
    from jaxposeidon import (
        _opacity_precompute, _opacities, _atmosphere, _chemistry, _clouds,
        _instruments, _geometry, _transmission, _parameters, _data, _priors,
        _constants, _species_data, core,
    )
    assert jaxposeidon.__version__ == "0.0.1.dev0"
    # All stub modules are importable but empty for v0 Phase 0.
    for mod in (_opacity_precompute, _opacities, _atmosphere, _chemistry,
                _clouds, _instruments, _geometry, _transmission, _parameters,
                _data, _priors, _constants, _species_data, core):
        assert mod is not None


def test_poseidon_oracle_available():
    """POSEIDON imports — required for all subsequent phase tests."""
    import POSEIDON
    from POSEIDON.core import (
        create_star, create_planet, define_model,
        read_opacities, make_atmosphere, compute_spectrum,
    )
    assert all(callable(f) for f in (
        create_star, create_planet, define_model,
        read_opacities, make_atmosphere, compute_spectrum,
    ))


def test_k2_18b_species_supported_in_poseidon():
    """K2-18 b target species are in POSEIDON's supported_species table.

    DMS = C2H6S, DMDS = C2H6S2 per POSEIDON naming convention.
    """
    from POSEIDON.supported_chemicals import supported_species

    targets = {"H2O", "CH4", "CO2", "CO", "NH3", "HCN",
               "OCS", "N2O", "CH3Cl", "CS2", "C2H6S"}
    missing = targets - set(supported_species.tolist())
    assert not missing, f"Missing K2-18b target species in POSEIDON: {missing}"
