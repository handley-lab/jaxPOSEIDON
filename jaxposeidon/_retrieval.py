"""Retrieval driver — Phase 10 (BlackJAX NSS integration).

v0 wires three pieces from earlier phases into a single sampler-ready
log-posterior callable:

    unit_cube → prior_transform → (compute_spectrum → bin → loglikelihood)

`ln_prior_TP` (the PT_penalty / Pelletier branch) is v1-deferred per
the plan's top-level deferred list and Phase 8 envelope.

The prior side (`_priors.prior_transform`) is a faithful port of
POSEIDON `retrieval.py:649-1008` filtered to v0 (uniform/gaussian/sine).
The likelihood side reuses `_compute_spectrum.compute_spectrum`,
`_instruments.bin_spectrum_to_data`, and `_data.loglikelihood`.

BlackJAX NSS execution itself is gated to v1 alongside the JAX-traceable
forward model — the numpy forward model used by v0 is not compatible
with the lax-traced inner loop NSS expects. The wiring here is
exercised by parity-check tests against a hand-replicated reference.

POSEIDON's MultiNest dispatch lives at `retrieval.py:146-172` and
`:1186-1187`; we replace it with the BlackJAX NSS wiring in v1 once
the forward model is JAX-traced.
"""

import numpy as np

from jaxposeidon._compute_spectrum import compute_spectrum
from jaxposeidon._data import loglikelihood
from jaxposeidon._instruments import bin_spectrum_to_data
from jaxposeidon._priors import prior_transform

_V0_REFERENCE_PARAMETERS = {"R_p_ref", "P_ref", "R_p_ref+P_ref"}


def make_loglikelihood(
    planet,
    star,
    model,
    opac,
    wl,
    data_properties,
    split_params,
    make_atmosphere,
    param_names,
    prior_types,
    prior_ranges,
    *,
    P,
    reference_parameter="R_p_ref",
    R_p_ref_fixed=None,
    log_P_ref_fixed=None,
    offsets_applied=None,
    error_inflation=None,
    N_params_cum=None,
):
    """Build a closure cube → log-posterior over the v0 retrieval envelope.

    Reference-parameter handling (per Phase 1 v0 envelope):
        - 'R_p_ref'       : physical_params = [R_p_ref]; P_ref fixed.
        - 'P_ref'         : physical_params = [log_P_ref]; R_p_ref fixed.
        - 'R_p_ref+P_ref' : physical_params = [R_p_ref, log_P_ref].

    `split_params` and `make_atmosphere` are passed in by the caller so
    this module does not import POSEIDON. They are typically:
        - `jaxposeidon._parameters.split_params`
        - `POSEIDON.core.make_atmosphere`

    `P` is the retrieval pressure grid; the caller supplies it
    explicitly so the driver does not hard-code POSEIDON conventions.
    """
    if reference_parameter not in _V0_REFERENCE_PARAMETERS:
        raise NotImplementedError(
            f"reference_parameter={reference_parameter!r} not in v0 "
            f"({sorted(_V0_REFERENCE_PARAMETERS)})"
        )
    if reference_parameter == "P_ref" and R_p_ref_fixed is None:
        raise ValueError("reference_parameter='P_ref' requires R_p_ref_fixed")
    if reference_parameter == "R_p_ref" and log_P_ref_fixed is None:
        raise ValueError("reference_parameter='R_p_ref' requires log_P_ref_fixed")

    def logp(unit_cube):
        physical = prior_transform(
            np.asarray(unit_cube),
            param_names,
            prior_types,
            prior_ranges,
        )
        (
            physical_params,
            PT_params,
            log_X_params,
            cloud_params,
            geometry_params,
            *rest,
        ) = split_params(physical, N_params_cum)
        # rest contains stellar, offsets, err_inflation, high_res, surface;
        # v0 uses offsets, err_inflation; the rest are no-ops.
        offset_params = rest[1] if len(rest) > 1 else np.array([])
        err_inflation_params = rest[2] if len(rest) > 2 else np.array([])

        if reference_parameter == "R_p_ref":
            R_p_ref = physical_params[0]
            P_ref = 10.0**log_P_ref_fixed
        elif reference_parameter == "P_ref":
            R_p_ref = R_p_ref_fixed
            P_ref = 10.0 ** physical_params[0]
        else:  # 'R_p_ref+P_ref'
            R_p_ref = physical_params[0]
            P_ref = 10.0 ** physical_params[1]
        atmosphere = make_atmosphere(
            planet,
            model,
            P,
            P_ref,
            R_p_ref,
            PT_params,
            log_X_params,
            cloud_params=cloud_params,
            geometry_params=geometry_params,
            constant_gravity=True,
        )
        spectrum = compute_spectrum(
            planet, star, model, atmosphere, opac, wl, spectrum_type="transmission"
        )
        ymodel = bin_spectrum_to_data(spectrum, wl, data_properties)
        ll = loglikelihood(
            ymodel,
            data_properties["ydata"],
            data_properties["err_data"],
            offset_params=offset_params,
            err_inflation_params=err_inflation_params,
            offsets_applied=offsets_applied,
            error_inflation=error_inflation,
            offset_start=data_properties["offset_start"],
            offset_end=data_properties["offset_end"],
            offset_1_start=data_properties["offset_1_start"],
            offset_1_end=data_properties["offset_1_end"],
            offset_2_start=data_properties["offset_2_start"],
            offset_2_end=data_properties["offset_2_end"],
            offset_3_start=data_properties["offset_3_start"],
            offset_3_end=data_properties["offset_3_end"],
        )
        return ll

    return logp


def run_NSS(logp, n_dim, n_live=200, key_seed=0):
    """Run BlackJAX NSS over the JAX log-posterior callable.

    *** v1: requires JAX-traceable forward model. ***

    POSEIDON's MultiNest sampler dispatch is at
    `retrieval.py:146-172`; BlackJAX NSS replaces it via the
    handley-lab fork, reusing the same parameter ordering and
    prior transform.
    """
    raise NotImplementedError(
        "BlackJAX NSS run loop is gated to v1 — the v0 forward model "
        "is a numpy port and not yet JAX-traceable. Once the forward "
        "model becomes lax-traced, blackjax.nss(...) will plug "
        "directly into `logp` here."
    )
