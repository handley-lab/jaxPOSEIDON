"""jaxposeidon — JAX-native port of POSEIDON's transmission forward model.

Status: Phase 1 (parameter/state layer).
"""

import jax as _jax

# Enable float64 globally — required for atol=1e-12 parameter/state parity
# with POSEIDON's float64 path. POSEIDON casts opacity arrays to float32
# explicitly where applicable; we mirror that locally, not globally.
_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.2.dev0"
