"""pytest configuration for jaxposeidon tests.

Synthetic HDF5 opacity fixtures are constructed here per the plan:
- Smoke tests (testing=True) only need a synthetic
  `opacity/Opacity_database_cia.hdf5`.
- Molecular tests (testing=False) additionally need a synthetic
  `opacity/Opacity_database_v1.3.hdf5` containing at least `H2O/log(P)`
  (POSEIDON reads the pressure grid from this group unconditionally —
  see `POSEIDON/POSEIDON/absorption.py:806-808`).

These fixtures are not yet implemented — Phase 3 (opacity preprocessing)
will populate them. For Phase 0 we rely on the user's real
`POSEIDON_input_data` directory being on PATH if any test that needs
molecular HDF5 is run.
"""

import os
import pytest


@pytest.fixture(scope="session")
def poseidon_input_data():
    """Path to POSEIDON's opacity database; required for non-testing-mode runs."""
    p = os.environ.get("POSEIDON_input_data")
    if not p:
        pytest.skip(
            "POSEIDON_input_data env var not set; skipping tests that need "
            "the opacity database. Phase 3 will add synthetic fixtures."
        )
    return p
