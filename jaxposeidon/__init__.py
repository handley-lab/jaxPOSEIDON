"""jaxposeidon — JAX-friendly port of POSEIDON's forward model.

Status: v1.0.0 — partial JAX-trace coverage. The v1-A through v1-E
JAX migrations have landed; seven hot-path modules (`_data`,
`_emission`, `_jax_transmission`, `_opacities`, `_parameters`,
`_priors`, `_stellar`) are fully JAX-pure, and eleven remain
grandfathered as v1.0.x follow-ups (see ``MISMATCHES.md`` →
"v1.0.0 source-grep grandfather list"). The BlackJAX NSS sampler
run consumes the jit-traceable surface today via the leaf kernels.

Public API:
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

__version__ = "1.0.0"

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
from jaxposeidon._setup_api import (
    create_planet,
    create_star,
    define_model,
    make_atmosphere,
    read_opacities,
    wl_grid_constant_R,
)

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
    "create_star",
    "create_planet",
    "define_model",
    "read_opacities",
    "make_atmosphere",
    "wl_grid_constant_R",
]
