"""pytest configuration for jaxposeidon tests.

Phase tests that need POSEIDON's HDF5 fixtures build them in-tempdir
per-module (see e.g. `tests/test_phase9_compute_spectrum.py`,
`tests/test_phase9_sweep.py`, `tests/test_phase10_retrieval.py` for
the H2-H2 / H2-He CIA pattern, and the inline `tmp_path` synthetic
`Opacity_database_v1.3.hdf5` build in `test_phase9_compute_spectrum`'s
molecular opacity test).

The session-scoped `poseidon_input_data` fixture below is for tests
that want the *real* opacity database; it skips when
`POSEIDON_input_data` is unset.
"""

import os
import pytest
import jax

jax.config.update("jax_enable_x64", True)

# POSEIDON.emission reads `block` and `thread` from os.environ at import
# time (CUDA grid sizing for the GPU planck path). Default them here so
# tests that import POSEIDON.emission don't KeyError on the lookup.
os.environ.setdefault("block", "32")
os.environ.setdefault("thread", "64")


@pytest.fixture(scope="session")
def poseidon_input_data():
    """Real POSEIDON opacity database path; skip when not provided."""
    p = os.environ.get("POSEIDON_input_data")
    if not p:
        pytest.skip(
            "POSEIDON_input_data env var not set; skipping tests that need "
            "the real opacity database. Synthetic HDF5 fixtures are built "
            "per-test where applicable."
        )
    return p
