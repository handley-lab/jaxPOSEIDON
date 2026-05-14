"""jaxposeidon — JAX-friendly port of POSEIDON's transmission forward model.

Status: v0 complete (Phases 0-10 approved; Phase 11 packaging in progress).
The v0 forward model is a numpy port validated against POSEIDON. The
JAX-traceable migration and the BlackJAX NSS sampler run are v1 work
items.

Public API (post-Phase 9):
    jaxposeidon.compute_spectrum
    jaxposeidon.check_atmosphere_physical
    jaxposeidon.bin_spectrum_to_data
    jaxposeidon.make_model_data
    jaxposeidon.compute_instrument_indices
    jaxposeidon.loglikelihood
    jaxposeidon.apply_offsets
    jaxposeidon.effective_error_sq
    jaxposeidon.prior_transform
    jaxposeidon.make_loglikelihood
    jaxposeidon.load_data       # thin POSEIDON wrapper
    jaxposeidon.init_instrument # thin POSEIDON wrapper
"""

import jax as _jax

# Enable float64 globally — required for atol=1e-12 parameter/state parity
# with POSEIDON's float64 path. POSEIDON casts opacity arrays to float32
# explicitly where applicable; we mirror that locally, not globally.
_jax.config.update("jax_enable_x64", True)

__version__ = "0.0.4"

from jaxposeidon._compute_spectrum import (
    check_atmosphere_physical,
    compute_spectrum,
)
from jaxposeidon._data import (
    apply_offsets,
    effective_error_sq,
    loglikelihood,
)
from jaxposeidon._instruments import (
    bin_spectrum_to_data,
    compute_instrument_indices,
    make_model_data,
)
from jaxposeidon._loaddata import init_instrument, load_data
from jaxposeidon._priors import prior_transform
from jaxposeidon._retrieval import make_loglikelihood

__all__ = [
    "compute_spectrum",
    "check_atmosphere_physical",
    "bin_spectrum_to_data",
    "make_model_data",
    "compute_instrument_indices",
    "loglikelihood",
    "apply_offsets",
    "effective_error_sq",
    "prior_transform",
    "make_loglikelihood",
    "load_data",
    "init_instrument",
]
