"""Real-JAX high-res sysrem log-likelihood kernel — transmission path.

Tests `_jax_high_res.loglikelihood_sysrem_kernel_jit` parity vs the
numpy `_high_res.loglikelihood_sysrem` for the transmission
`star_spectrum=None` call pattern (the most common high-res
use-case: hot Jupiter + cross-correlation).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxposeidon import _high_res
from jaxposeidon._jax_high_res import loglikelihood_sysrem_kernel_jit


def _fixture(N_orders=2, N_phi=8, N_pix=20, with_b=False, with_uncertainties=False, seed=0):
    rng = np.random.default_rng(seed)
    N_wl = 80
    wl = np.linspace(1.0, 5.0, N_wl)
    wl_grid = np.tile(np.linspace(2.0, 4.0, N_pix)[None, :], (N_orders, 1))
    planet_spectrum = 1e-3 * (1 + 0.1 * np.sin(2 * np.pi * wl))
    phi = np.linspace(-0.05, 0.05, N_phi)
    V_bary = np.full(N_phi, 5.0)
    residuals = 1e-4 * rng.standard_normal((N_orders, N_phi, N_pix))
    # Bs: per-order frame-projection matrix shape (N_phi, N_phi) — applied
    # as `B @ models_shifted` where models_shifted is (N_phi, N_pix).
    Bs = 0.01 * rng.standard_normal((N_orders, N_phi, N_phi))
    transit_weight = np.linspace(1.0, 0.95, N_phi)
    uncertainties = (
        rng.uniform(0.9, 1.1, (N_orders, N_phi, N_pix))
        if with_uncertainties
        else np.ones((N_orders, N_phi, N_pix))
    )
    max_transit_depth = float(np.max(1 - transit_weight))
    return dict(
        V_sys=10.0,
        K_p=180.0,
        d_phi=0.001,
        a=0.8,
        b=1.2 if with_b else None,
        wl=wl,
        planet_spectrum=planet_spectrum,
        wl_grid=wl_grid,
        residuals=residuals,
        Bs=Bs,
        phi=phi,
        V_bary=V_bary,
        uncertainties=uncertainties,
        transit_weight=transit_weight,
        max_transit_depth=max_transit_depth,
    )


@pytest.mark.parametrize(
    "label,with_b,with_uncertainties",
    [
        ("no_uncertainties_no_b", False, False),
        ("with_uncertainties_no_b", False, True),
        # Skip ("with_b_no_uncertainties", True, False) — invalid in POSEIDON
        # (b!=None branch dereferences uncertainties[i,j]).
        ("with_b_and_uncertainties", True, True),
    ],
)
def test_loglikelihood_sysrem_jit_parity_transmission(label, with_b, with_uncertainties):
    cfg = _fixture(with_b=with_b, with_uncertainties=with_uncertainties)

    # Build POSEIDON data dict
    data = {
        "wl_grid": cfg["wl_grid"],
        "residuals": cfg["residuals"],
        "Bs": cfg["Bs"],
        "phi": cfg["phi"],
        "V_bary": cfg["V_bary"],
        "transit_weight": cfg["transit_weight"],
    }
    if with_uncertainties:
        data["uncertainties"] = cfg["uncertainties"]
    # Numpy reference
    ll_t = _high_res.loglikelihood_sysrem(
        cfg["V_sys"], cfg["K_p"], cfg["d_phi"], cfg["a"], cfg["b"],
        cfg["wl"], cfg["planet_spectrum"], data, star_spectrum=None,
    )

    # JAX kernel
    ll_o = loglikelihood_sysrem_kernel_jit(
        jnp.float64(cfg["V_sys"]),
        jnp.float64(cfg["K_p"]),
        jnp.float64(cfg["d_phi"]),
        jnp.float64(cfg["a"]),
        jnp.float64(cfg["b"]) if cfg["b"] is not None else jnp.float64(1.0),
        jnp.asarray(cfg["wl"]),
        jnp.asarray(cfg["planet_spectrum"]),
        jnp.asarray(cfg["wl_grid"]),
        jnp.asarray(cfg["residuals"]),
        jnp.asarray(cfg["Bs"]),
        jnp.asarray(cfg["phi"]),
        jnp.asarray(cfg["V_bary"]),
        jnp.asarray(cfg["uncertainties"]),
        jnp.asarray(cfg["transit_weight"]),
        jnp.zeros_like(cfg["planet_spectrum"]),  # star_spectrum unused
        jnp.float64(cfg["max_transit_depth"]),
        use_star=False,
        use_b=with_b,
        use_uncertainties=with_uncertainties,
    )

    # rtol relaxed from default 1e-13 → 1e-12 because vmap-over-orders +
    # einsum reductions reorder the sum vs POSEIDON's nested Python
    # loops. Documented in MISMATCHES.md.
    np.testing.assert_allclose(float(ll_o), float(ll_t), rtol=1e-12, atol=0)


def test_loglikelihood_sysrem_jit_is_jit_traceable():
    cfg = _fixture(with_uncertainties=True)

    @jax.jit
    def _f(V_sys, K_p, d_phi, a, planet_spectrum):
        return loglikelihood_sysrem_kernel_jit(
            V_sys, K_p, d_phi, a, jnp.float64(1.0),
            jnp.asarray(cfg["wl"]), planet_spectrum,
            jnp.asarray(cfg["wl_grid"]), jnp.asarray(cfg["residuals"]),
            jnp.asarray(cfg["Bs"]), jnp.asarray(cfg["phi"]),
            jnp.asarray(cfg["V_bary"]), jnp.asarray(cfg["uncertainties"]),
            jnp.asarray(cfg["transit_weight"]),
            jnp.zeros_like(cfg["planet_spectrum"]),
            jnp.float64(cfg["max_transit_depth"]),
            use_star=False, use_b=False, use_uncertainties=True,
        )

    ll = _f(
        jnp.float64(cfg["V_sys"]), jnp.float64(cfg["K_p"]),
        jnp.float64(cfg["d_phi"]), jnp.float64(cfg["a"]),
        jnp.asarray(cfg["planet_spectrum"]),
    )
    assert jnp.isfinite(ll)


def test_grad_through_jit_loglikelihood_sysrem_via_K_p():
    """jax.grad through K_p (the orbital RV semi-amplitude) is the
    canonical retrieval-parameter gradient — must flow."""
    cfg = _fixture(with_uncertainties=True)

    def _loss(K_p):
        return loglikelihood_sysrem_kernel_jit(
            jnp.float64(cfg["V_sys"]), K_p,
            jnp.float64(cfg["d_phi"]), jnp.float64(cfg["a"]),
            jnp.float64(1.0),
            jnp.asarray(cfg["wl"]), jnp.asarray(cfg["planet_spectrum"]),
            jnp.asarray(cfg["wl_grid"]), jnp.asarray(cfg["residuals"]),
            jnp.asarray(cfg["Bs"]), jnp.asarray(cfg["phi"]),
            jnp.asarray(cfg["V_bary"]), jnp.asarray(cfg["uncertainties"]),
            jnp.asarray(cfg["transit_weight"]),
            jnp.zeros_like(cfg["planet_spectrum"]),
            jnp.float64(cfg["max_transit_depth"]),
            use_star=False, use_b=False, use_uncertainties=True,
        )

    K0 = jnp.float64(cfg["K_p"])
    g = jax.grad(jax.jit(_loss))(K0)
    assert jnp.isfinite(g)
    assert g != 0.0
    # FD cross-check
    h = 1e-2
    fd = (_loss(K0 + h) - _loss(K0 - h)) / (2 * h)
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-4, atol=1e-30)


def test_grad_through_jit_loglikelihood_sysrem_via_planet_spectrum():
    """jax.grad through planet_spectrum — the forward-model output.
    This is the key gradient for joint retrieval (spectrum params
    flow through TRIDENT / emission_Toon into planet_spectrum, which
    then feeds the likelihood)."""
    cfg = _fixture(with_uncertainties=True)

    def _loss(planet_spectrum):
        return loglikelihood_sysrem_kernel_jit(
            jnp.float64(cfg["V_sys"]), jnp.float64(cfg["K_p"]),
            jnp.float64(cfg["d_phi"]), jnp.float64(cfg["a"]),
            jnp.float64(1.0),
            jnp.asarray(cfg["wl"]), planet_spectrum,
            jnp.asarray(cfg["wl_grid"]), jnp.asarray(cfg["residuals"]),
            jnp.asarray(cfg["Bs"]), jnp.asarray(cfg["phi"]),
            jnp.asarray(cfg["V_bary"]), jnp.asarray(cfg["uncertainties"]),
            jnp.asarray(cfg["transit_weight"]),
            jnp.zeros_like(cfg["planet_spectrum"]),
            jnp.float64(cfg["max_transit_depth"]),
            use_star=False, use_b=False, use_uncertainties=True,
        )

    g = jax.grad(jax.jit(_loss))(jnp.asarray(cfg["planet_spectrum"]))
    assert g.shape == cfg["planet_spectrum"].shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)


def test_make_jaxpr_loglikelihood_sysrem_succeeds():
    cfg = _fixture(with_uncertainties=True)

    def _f(planet_spectrum):
        return loglikelihood_sysrem_kernel_jit(
            jnp.float64(cfg["V_sys"]), jnp.float64(cfg["K_p"]),
            jnp.float64(cfg["d_phi"]), jnp.float64(cfg["a"]),
            jnp.float64(1.0),
            jnp.asarray(cfg["wl"]), planet_spectrum,
            jnp.asarray(cfg["wl_grid"]), jnp.asarray(cfg["residuals"]),
            jnp.asarray(cfg["Bs"]), jnp.asarray(cfg["phi"]),
            jnp.asarray(cfg["V_bary"]), jnp.asarray(cfg["uncertainties"]),
            jnp.asarray(cfg["transit_weight"]),
            jnp.zeros_like(cfg["planet_spectrum"]),
            jnp.float64(cfg["max_transit_depth"]),
            use_star=False, use_b=False, use_uncertainties=True,
        )

    jaxpr = jax.make_jaxpr(_f)(jnp.asarray(cfg["planet_spectrum"]))
    assert jaxpr is not None
    assert "pure_callback" not in str(jaxpr)
