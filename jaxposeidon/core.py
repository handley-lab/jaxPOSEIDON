"""Public API mirror, matching POSEIDON's `core` module surface where v0 ports it.

This module re-exports the v0 jaxposeidon API for callers that want
`from jaxposeidon.core import compute_spectrum, ...`. POSEIDON itself
exposes its public functions through `POSEIDON.core`, so we mirror the
import path for parity.

The functions themselves are defined in their respective phase modules:
    compute_spectrum, check_atmosphere_physical → _compute_spectrum
    bin_spectrum_to_data, make_model_data,
    compute_instrument_indices              → _instruments
    loglikelihood, apply_offsets,
    effective_error_sq                      → _data
    prior_transform                         → _priors
    make_loglikelihood                      → _retrieval
    load_data, init_instrument              → _loaddata (POSEIDON shims)

POSEIDON-side public-API entry points that are NOT yet in v0 because
they live behind v1-deferred work (Mie / eddysed / emission / stellar /
LBL etc.) are intentionally omitted; calling them through
`jaxposeidon.core` will raise `ImportError` and the caller is expected
to use POSEIDON directly until those phases land.
"""

from jaxposeidon._compute_spectrum import (
    compute_spectrum, check_atmosphere_physical,
)
from jaxposeidon._instruments import (
    bin_spectrum_to_data, make_model_data, compute_instrument_indices,
)
from jaxposeidon._data import (
    loglikelihood, apply_offsets, effective_error_sq,
)
from jaxposeidon._priors import prior_transform
from jaxposeidon._retrieval import make_loglikelihood
from jaxposeidon._loaddata import load_data, init_instrument

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
