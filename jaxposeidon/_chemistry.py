"""Chemistry layer — v0 stub.

The plan's Phase 2 dispatch covers free / isochem chemistry directly
in `_atmosphere.compute_X_isochem_1D` and `_atmosphere.add_bulk_component`.
POSEIDON's separate `chemistry.py` is concerned with FastChem-based
equilibrium chemistry grids; that is deferred to v1.

This module exists for import-compatibility with the planned package
architecture; equilibrium-chemistry entry points raise NotImplementedError.
"""


def interpolate_log_X_grid(*args, **kwargs):
    """Equilibrium chemistry interpolation — deferred to v1.

    Mirrors POSEIDON `chemistry.py:interpolate_log_X_grid` (FastChem grid
    look-up), used by `atmosphere.profiles(...)` when
    `X_profile='chem_eq'`. v0 does not support equilibrium chemistry.
    """
    raise NotImplementedError(
        "Equilibrium chemistry (chem_eq) is deferred to v1; v0 supports "
        "X_profile='isochem' free chemistry only."
    )


def load_chemistry_grid(*args, **kwargs):
    """FastChem grid loader — deferred to v1."""
    raise NotImplementedError("FastChem chemistry grid loading is deferred to v1.")
