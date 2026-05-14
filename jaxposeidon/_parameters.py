"""Parameter hot-path — JAX-pure split/unpack of already-constructed
parameter vectors.

Setup-time logic (`assign_free_params`, `assert_v0_model_config`,
the `V0_*` whitelist constants) lives in `_parameter_setup.py`
(allow-listed setup-only module per `CLAUDE.md`). This file is the
hot-path companion called inside `make_loglikelihood`'s closure and
must stay JAX-pure under the v1 source-grep gate.

Back-compat re-exports: `assign_free_params`, `assert_v0_model_config`,
and the `V0_*` constants are re-exported here so existing v0 tests
and external callers (`jaxposeidon._parameters.assign_free_params`,
etc.) continue to work without modification.
"""

# Back-compat re-exports from the setup-only module.
from jaxposeidon._parameter_setup import (  # noqa: F401
    V0_CLOUD_DIMS,
    V0_CLOUD_MODELS,
    V0_CLOUD_TYPES,
    V0_ERROR_INFLATIONS,
    V0_OFFSETS,
    V0_PT_PROFILES,
    V0_REFERENCE_PARAMETERS,
    V0_X_PROFILES,
    assert_v0_model_config,
    assign_free_params,
)


# ---------------------------------------------------------------------------
# split_params — JAX-hot-path companion (parameters.py:1157-1224 port)
# ---------------------------------------------------------------------------
def split_params(params_drawn, N_params_cumulative):
    """Split a flat parameter vector into POSEIDON's typed groups.

    Returns ten arrays in POSEIDON's canonical order:
        physical_drawn, PT_drawn, log_X_drawn, clouds_drawn, geometry_drawn,
        stellar_drawn, offsets_drawn, err_inflation_drawn, high_res_drawn,
        surface_drawn.
    """
    return (
        params_drawn[0 : N_params_cumulative[0]],
        params_drawn[N_params_cumulative[0] : N_params_cumulative[1]],
        params_drawn[N_params_cumulative[1] : N_params_cumulative[2]],
        params_drawn[N_params_cumulative[2] : N_params_cumulative[3]],
        params_drawn[N_params_cumulative[3] : N_params_cumulative[4]],
        params_drawn[N_params_cumulative[4] : N_params_cumulative[5]],
        params_drawn[N_params_cumulative[5] : N_params_cumulative[6]],
        params_drawn[N_params_cumulative[6] : N_params_cumulative[7]],
        params_drawn[N_params_cumulative[7] : N_params_cumulative[8]],
        params_drawn[N_params_cumulative[8] : N_params_cumulative[9]],
    )
