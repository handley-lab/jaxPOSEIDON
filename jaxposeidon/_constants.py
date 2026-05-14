"""POSEIDON physical constants — v0 thin re-export.

POSEIDON ships its physical constants in `POSEIDON.constants` (`R_J`,
`M_J`, `R_Sun`, etc.). For v0 the jaxposeidon test setup uses
POSEIDON's values directly via `from POSEIDON.constants import ...`,
so this module is a stable namespace for any constants we eventually
want to extract at build time (currently none — `_species_data.py`
already holds the species masses table extracted at Phase 2).

Phase 3+ did not require a build-time constants extraction beyond
species masses; this module remains as a stable import path
documented in the plan's architecture tree.
"""
