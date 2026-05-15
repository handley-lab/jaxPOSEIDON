"""Offsets, error inflation, Gaussian likelihood — v1 JAX port.

Faithful port of POSEIDON `retrieval.py:1065-1183` (the `LogLikelihood`
body's likelihood-arithmetic section), filtered to the v0 envelope.
Hot-path numeric kernels use `jnp` so the call site is jit-able when
the dispatch tags (`offsets_applied`, `error_inflation`) are static.

v0 envelope:
- `error_inflation in {None, 'Line15', 'Piette20', 'Line15+Piette20'}`
- `offsets_applied in {None, 'single_dataset', 'two_datasets',
  'three_datasets'}`
- spectroscopic-only data (photometric handled upstream)

Reject NaN-bearing spectra with the same -1e100 sentinel POSEIDON uses
(`retrieval.py:1066-1072`).
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402


def _is_scalar_zero_marker(v):
    return isinstance(v, (int, float)) and v == 0


def apply_offsets(
    ydata,
    offset_params,
    offsets_applied,
    offset_start,
    offset_end,
    offset_1_start=0,
    offset_1_end=0,
    offset_2_start=0,
    offset_2_end=0,
    offset_3_start=0,
    offset_3_end=0,
):
    """Apply per-dataset ppm-scale offsets to the observed transit depths.

    Mirrors POSEIDON `retrieval.py:1124-1174`. `offset_params` is the
    `offset_params` slice returned by `split_params`. `offsets_applied`
    selects how many independent offsets are present.
    """
    if offsets_applied is None:
        return ydata
    ydata_adj = jnp.asarray(ydata, dtype=jnp.float64)
    offset_params = jnp.asarray(offset_params, dtype=jnp.float64)

    if offsets_applied == "single_dataset":
        if _is_scalar_zero_marker(offset_1_start):
            ydata_adj = ydata_adj.at[offset_start:offset_end].add(
                -offset_params[0] * 1e-6
            )
        else:
            for n in range(len(offset_1_start)):
                ydata_adj = ydata_adj.at[offset_1_start[n] : offset_1_end[n]].add(
                    -offset_params[0] * 1e-6
                )
    elif offsets_applied == "two_datasets":
        if _is_scalar_zero_marker(offset_1_start):
            ydata_adj = ydata_adj.at[offset_start[0] : offset_end[0]].add(
                -offset_params[0] * 1e-6
            )
            ydata_adj = ydata_adj.at[offset_start[1] : offset_end[1]].add(
                -offset_params[1] * 1e-6
            )
        else:
            for n in range(len(offset_1_start)):
                ydata_adj = ydata_adj.at[offset_1_start[n] : offset_1_end[n]].add(
                    -offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj = ydata_adj.at[offset_2_start[m] : offset_2_end[m]].add(
                    -offset_params[1] * 1e-6
                )
    elif offsets_applied == "three_datasets":
        if _is_scalar_zero_marker(offset_1_start):
            ydata_adj = ydata_adj.at[offset_start[0] : offset_end[0]].add(
                -offset_params[0] * 1e-6
            )
            ydata_adj = ydata_adj.at[offset_start[1] : offset_end[1]].add(
                -offset_params[1] * 1e-6
            )
            ydata_adj = ydata_adj.at[offset_start[2] : offset_end[2]].add(
                -offset_params[2] * 1e-6
            )
        else:
            for n in range(len(offset_1_start)):
                ydata_adj = ydata_adj.at[offset_1_start[n] : offset_1_end[n]].add(
                    -offset_params[0] * 1e-6
                )
            for m in range(len(offset_2_start)):
                ydata_adj = ydata_adj.at[offset_2_start[m] : offset_2_end[m]].add(
                    -offset_params[1] * 1e-6
                )
            for s in range(len(offset_3_start)):
                ydata_adj = ydata_adj.at[offset_3_start[s] : offset_3_end[s]].add(
                    -offset_params[2] * 1e-6
                )
    else:
        raise NotImplementedError(f"offsets_applied={offsets_applied!r} not in v0")
    return ydata_adj


def effective_error_sq(
    err_data, ymodel, err_inflation_params, error_inflation, norm_log_default=0.0
):
    """(err_eff_sq, norm_log) per POSEIDON `retrieval.py:1097-1110`."""
    err_data = jnp.asarray(err_data, dtype=jnp.float64)
    ymodel = jnp.asarray(ymodel, dtype=jnp.float64)
    if error_inflation is None:
        return err_data * err_data, norm_log_default
    err_inflation_params = jnp.asarray(err_inflation_params, dtype=jnp.float64)
    if error_inflation == "Line15":
        err_eff_sq = err_data * err_data + 10.0 ** err_inflation_params[0]
    elif error_inflation == "Piette20":
        err_eff_sq = err_data * err_data + (err_inflation_params[0] * ymodel) ** 2
    elif error_inflation == "Line15+Piette20":
        err_eff_sq = (
            err_data * err_data
            + 10.0 ** err_inflation_params[0]
            + (err_inflation_params[1] * ymodel) ** 2
        )
    else:
        raise NotImplementedError(f"error_inflation={error_inflation!r} not in v0")
    norm_log = (-0.5 * jnp.log(2.0 * jnp.pi * err_eff_sq)).sum()
    return err_eff_sq, norm_log


def loglikelihood(
    ymodel,
    ydata,
    err_data,
    *,
    offset_params=(),
    err_inflation_params=(),
    offsets_applied=None,
    error_inflation=None,
    offset_start=0,
    offset_end=0,
    offset_1_start=0,
    offset_1_end=0,
    offset_2_start=0,
    offset_2_end=0,
    offset_3_start=0,
    offset_3_end=0,
    norm_log_default=None,
    ln_prior_TP=0.0,
):
    """Gaussian likelihood with optional offsets + error inflation.

    Mirrors POSEIDON `retrieval.py:1065-1183` for the v0 envelope.
    Returns `-1e100` if `ymodel` contains NaN (POSEIDON's unphysical-
    spectrum sentinel at `retrieval.py:1066-1072`).
    """
    ymodel = jnp.asarray(ymodel, dtype=jnp.float64)
    err_data = jnp.asarray(err_data, dtype=jnp.float64)

    if error_inflation is None and norm_log_default is None:
        norm_log_default = (-0.5 * jnp.log(2.0 * jnp.pi * err_data * err_data)).sum()
    elif norm_log_default is None:
        norm_log_default = 0.0

    err_eff_sq, norm_log = effective_error_sq(
        err_data,
        ymodel,
        jnp.asarray(err_inflation_params)
        if len(err_inflation_params)
        else jnp.zeros(0),
        error_inflation,
        norm_log_default,
    )

    ydata_adjusted = apply_offsets(
        ydata,
        jnp.asarray(offset_params) if len(offset_params) else jnp.zeros(0),
        offsets_applied,
        offset_start,
        offset_end,
        offset_1_start,
        offset_1_end,
        offset_2_start,
        offset_2_end,
        offset_3_start,
        offset_3_end,
    )

    ll = (-0.5 * (ymodel - ydata_adjusted) ** 2 / err_eff_sq).sum() + norm_log
    ll = ll + ln_prior_TP
    nan_sentinel = jnp.float64(-1.0e100)
    return jnp.where(jnp.any(jnp.isnan(ymodel)), nan_sentinel, ll)
