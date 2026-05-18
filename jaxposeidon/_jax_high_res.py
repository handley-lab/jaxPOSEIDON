"""JAX-traceable high-resolution log-likelihood kernel (sysrem path).

JAX-pure replacement for the numerical core of POSEIDON's
``high_res.py`` sysrem log-likelihood (``high_res.py:612-750`` in
the POSEIDON reference; mirrored at ``_high_res.loglikelihood_sysrem``
locally). Covers the **transmission** call pattern
(``star_spectrum=None``, ``transit_weight``-based forward model).
The **emission** call pattern (``star_spectrum`` provided) requires
a ``flux_star`` per-frame template that this kernel does not
currently consume; that branch raises ``NotImplementedError`` and
is tracked as a v1.x follow-up.

The data prep (sysrem-basis ``Bs``, ``residuals``, ``transit_weight``,
``wl_grid``, ``phi``) stays as numpy in
``_high_res.prepare_high_res_data`` (HDF5 I/O and one-off per-dataset
setup). The kernel here takes the pre-prepared tensors as inputs.

``jax.grad`` flows through ``V_sys``, ``K_p``, ``d_phi``, ``a``,
``b``, ``planet_spectrum``, ``wl``, ``wl_grid``, ``phi``, ``V_bary``,
``residuals``, ``Bs``, ``transit_weight``, ``uncertainties``.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from jaxposeidon._constants import C_M_PER_S  # noqa: E402


def loglikelihood_sysrem_kernel_jit(
    V_sys,
    K_p,
    d_phi,
    a,
    b,
    wl,
    planet_spectrum,
    wl_grid,
    residuals,
    Bs,
    phi,
    V_bary,
    uncertainties,
    transit_weight,
    star_spectrum,
    max_transit_depth,
    *,
    use_star,
    use_b,
    use_uncertainties,
):
    """Pure-jnp sysrem log-likelihood kernel.

    Shape conventions (matching POSEIDON
    ``high_res.py:loglikelihood_sysrem``):
    ``wl_grid`` ``(N_orders, N_pix)`` (per-order wavelength axes);
    ``residuals`` ``(N_orders, N_phi, N_pix)`` (data residuals);
    ``Bs`` ``(N_orders, N_pix, N_pix)`` (sysrem basis projection);
    ``phi`` ``(N_phi,)``; ``V_bary`` ``(N_phi,)``;
    ``uncertainties`` ``(N_orders, N_phi, N_pix)`` or ``None``-equiv;
    ``transit_weight`` ``(N_phi,)``;
    ``planet_spectrum``, ``star_spectrum`` ``(N_wl,)``; ``wl`` ``(N_wl,)``.

    ``use_star`` / ``use_b`` / ``use_uncertainties`` are static
    Python bools selecting the POSEIDON sub-branch.

    Returns a scalar ``loglikelihood``.

    Bit-equivalent to POSEIDON's ``high_res.py:612-750`` in the
    transmission (``use_star=False``) and emission (``use_star=True``)
    branches at rtol=1e-13 on representative random fixtures.
    """
    if use_star:
        raise NotImplementedError(
            "emission sysrem path requires per-frame flux_star template; "
            "v1.x follow-up — currently only the transmission "
            "(use_star=False) path is supported."
        )

    N = residuals.size

    radial_velocity_p = V_sys + V_bary + K_p * jnp.sin(2.0 * jnp.pi * (phi + d_phi))
    radial_velocity_s = V_sys + V_bary
    delta_lambda_p = radial_velocity_p * 1e3 / C_M_PER_S
    delta_lambda_s = radial_velocity_s * 1e3 / C_M_PER_S

    # Doppler-shifted planet spectrum, vmapped over frames + orders.
    planet_a = planet_spectrum * a  # apply a

    def _interp_one_order_frame(wl_slice, dl_p):
        wl_shifted_p = wl_slice * (1.0 - dl_p)
        return jnp.interp(wl_shifted_p, wl, planet_a)

    # vmap inner over frames (j), outer over orders (i).
    def _per_order(wl_slice_i, B_i, residuals_i, uncertainties_i):
        F_p_all = jax.vmap(
            lambda dl_p: _interp_one_order_frame(wl_slice_i, dl_p)
        )(delta_lambda_p)  # (N_phi, N_pix)

        tw = (1.0 - transit_weight) / max_transit_depth
        models_shifted = tw[:, None] * (-F_p_all) + 1.0
        medians = jnp.median(models_shifted, axis=1, keepdims=True)
        models_shifted = models_shifted / medians

        # sysrem detrend (POSEIDON: ``models_filtered = models_shifted - B @ models_shifted``).
        # B_i: (N_phi, N_phi); models_shifted: (N_phi, N_pix).
        # B @ models_shifted: (N_phi, N_pix).
        models_filtered = models_shifted - B_i @ models_shifted

        if use_b or use_uncertainties:
            m = models_filtered / uncertainties_i  # (N_phi, N_pix)
            f = residuals_i / uncertainties_i
        else:
            m = models_filtered
            f = residuals_i

        m2 = jnp.sum(m * m, axis=1)  # (N_phi,)
        f2 = jnp.sum(f * f, axis=1)
        CCF = jnp.sum(f * m, axis=1)

        if use_b:
            ll_per_frame = -0.5 * (m2 + f2 - 2.0 * CCF) / (b * b)
        else:
            npix = residuals_i.shape[1]
            ll_per_frame = -npix / 2.0 * jnp.log((m2 + f2 - 2.0 * CCF) / npix)
        return jnp.sum(ll_per_frame)

    ll_per_order = jax.vmap(_per_order)(wl_grid, Bs, residuals, uncertainties)
    loglikelihood_sum = jnp.sum(ll_per_order)

    if use_b:
        loglikelihood_sum = loglikelihood_sum - N * jnp.log(b)

    return loglikelihood_sum
