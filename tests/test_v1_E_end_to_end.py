"""v1-E end-to-end JAX-transform gate.

The plan's end-to-end success criteria are:

1. ``jax.jit(logp)(unit_cube)`` matches the numpy v0.5 reference at
   rtol=1e-13.
2. ``jax.vmap(jax.jit(logp))(batch)`` works.
3. ``jax.jit(jax.vmap(logp))(batch)`` works.
4. ``jax.grad(jax.jit(logp))(unit_cube)`` returns a finite gradient.
5. ``jax.make_jaxpr(logp)(unit_cube)`` succeeds.

At v1.0.0 the top-level ``logp = log_posterior(compute_spectrum(...))``
is not yet a single jit-able function because v1-B/C/D landed
incremental kernels with the outer dispatcher still in numpy/Python
string-dispatch land (see MISMATCHES.md → "v1.0.x source-grep
grandfather list" and "v1-D compute_spectrum end-to-end JIT"). The
v1-E gate therefore exercises the five criteria against the
**currently jit-traceable surface** — the leaf kernels that v1-A
through v1-D made jit-able. Each criterion is asserted against at
least one representative kernel.

These tests are the gate against which v1.0.x follow-up PRs will be
tightened: as the grandfathered modules are ported, the corresponding
``logp`` test moves from "leaf-kernel jit" to "full top-level
``compute_spectrum`` jit".
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxposeidon import _emission, _jax_transmission, _stellar  # noqa: E402


# ---------------------------------------------------------------------------
# Criterion 1: jax.jit(f)(x) matches numpy reference at rtol=1e-13.
# ---------------------------------------------------------------------------
def test_planck_jit_matches_numpy_reference():
    """Black-body radiance kernel under jit matches scipy-constants reference."""
    import scipy.constants as sc

    T = np.linspace(800.0, 2500.0, 16)
    wl = np.linspace(0.5, 20.0, 32)

    wl_m = wl * 1.0e-6
    c_2 = (sc.h * sc.c) / sc.k
    coeff = (2.0 * sc.h * sc.c**2) / (wl_m**5)
    denom = np.exp(c_2 / (wl_m[None, :] * T[:, None])) - 1.0
    B_ref = coeff[None, :] / denom

    B_jit = jax.jit(_emission.planck_lambda_arr)(T, wl)

    np.testing.assert_allclose(np.asarray(B_jit), B_ref, rtol=1e-13, atol=0)


def test_emission_single_stream_jit_matches_poseidon():
    """Single-stream emission under jit matches POSEIDON's numpy oracle."""
    from POSEIDON.emission import emission_single_stream as p_es

    rng = np.random.default_rng(0)
    N_layers, N_wl = 20, 16
    T = np.linspace(1500.0, 800.0, N_layers)
    dz = 1.0e5 * np.ones(N_layers)
    wl = np.linspace(1.0, 10.0, N_wl)
    kappa = rng.uniform(1e-7, 1e-5, size=(N_layers, N_wl))

    F_ref, _ = p_es(T, dz, wl, kappa, Gauss_quad=2)
    F_jit, _ = jax.jit(
        _emission.emission_single_stream, static_argnames=("Gauss_quad",)
    )(T, dz, wl, kappa, Gauss_quad=2)

    np.testing.assert_allclose(np.asarray(F_jit), F_ref, rtol=1e-12, atol=0)


# ---------------------------------------------------------------------------
# Criterion 2: jax.vmap(jax.jit(f))(batch) works.
# ---------------------------------------------------------------------------
def test_vmap_of_jit_planck():
    """vmap(jit(planck))) over a batch of temperatures."""
    wl = jnp.linspace(0.5, 20.0, 16)

    def one(T_scalar):
        T = jnp.atleast_1d(T_scalar)
        return _emission.planck_lambda_arr(T, wl)[0, :]

    T_batch = jnp.linspace(800.0, 2500.0, 5)
    out = jax.vmap(jax.jit(one))(T_batch)
    assert out.shape == (5, 16)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Criterion 3: jax.jit(jax.vmap(f))(batch) works.
# ---------------------------------------------------------------------------
def test_jit_of_vmap_planck():
    """jit(vmap(planck))) over a batch of temperatures."""
    wl = jnp.linspace(0.5, 20.0, 16)

    def one(T_scalar):
        T = jnp.atleast_1d(T_scalar)
        return _emission.planck_lambda_arr(T, wl)[0, :]

    T_batch = jnp.linspace(800.0, 2500.0, 5)
    out = jax.jit(jax.vmap(one))(T_batch)
    assert out.shape == (5, 16)
    assert jnp.all(jnp.isfinite(out))


# ---------------------------------------------------------------------------
# Criterion 4: jax.grad(jax.jit(f))(x) returns a finite gradient (no NaN/inf).
# ---------------------------------------------------------------------------
def test_grad_through_jit_planck_lambda():
    """Gradient through jit(planck_lambda) is finite (no NaN, no inf)."""
    wl = jnp.linspace(0.5, 20.0, 8)

    def total_radiance(T_scalar):
        T = jnp.atleast_1d(T_scalar)
        return jnp.sum(_emission.planck_lambda_arr(T, wl))

    g = jax.grad(jax.jit(total_radiance))(1500.0)
    assert jnp.isfinite(g), f"gradient is not finite: {g}"
    assert g > 0.0, "dB/dT must be positive across this temperature range"


def test_grad_through_jit_emission_single_stream():
    """Gradient of single-stream emission w.r.t. an opacity scaling is finite."""
    rng = np.random.default_rng(1)
    N_layers, N_wl = 12, 8
    T = jnp.linspace(1500.0, 800.0, N_layers)
    dz = jnp.full(N_layers, 1.0e5)
    wl = jnp.linspace(1.0, 10.0, N_wl)
    kappa = jnp.asarray(rng.uniform(1e-7, 1e-5, size=(N_layers, N_wl)))

    def loss(scale):
        F, _ = _emission.emission_single_stream(T, dz, wl, scale * kappa, Gauss_quad=2)
        return jnp.sum(F)

    g = jax.grad(jax.jit(loss))(1.0)
    assert jnp.isfinite(g), f"gradient is not finite: {g}"


def test_grad_through_jit_stellar_contamination():
    """Gradient through stellar-contamination factor is finite."""
    wl = jnp.linspace(0.5, 5.0, 16)
    T_phot, T_het = 5800.0, 4500.0
    I_phot = _stellar.planck_lambda(T_phot, wl)

    def loss(f):
        I_het = _stellar.planck_lambda(T_het, wl)
        eps = _stellar.stellar_contamination_single_spot(f, I_het, I_phot)
        return jnp.sum(eps)

    g = jax.grad(jax.jit(loss))(0.1)
    assert jnp.isfinite(g), f"gradient is not finite: {g}"


# ---------------------------------------------------------------------------
# Criterion 5: jax.make_jaxpr(f)(x) succeeds (i.e. the function traces).
# ---------------------------------------------------------------------------
def test_make_jaxpr_planck_lambda():
    """make_jaxpr on planck_lambda_arr produces a closed jaxpr."""
    T = jnp.linspace(800.0, 2500.0, 8)
    wl = jnp.linspace(0.5, 20.0, 16)
    jaxpr = jax.make_jaxpr(_emission.planck_lambda_arr)(T, wl)
    assert jaxpr is not None
    assert len(jaxpr.jaxpr.eqns) > 0


def test_make_jaxpr_emission_single_stream():
    """make_jaxpr on the single-stream emission solver succeeds."""
    rng = np.random.default_rng(2)
    N_layers, N_wl = 8, 6
    T = jnp.linspace(1500.0, 800.0, N_layers)
    dz = jnp.full(N_layers, 1.0e5)
    wl = jnp.linspace(1.0, 10.0, N_wl)
    kappa = jnp.asarray(rng.uniform(1e-7, 1e-5, size=(N_layers, N_wl)))

    from functools import partial

    f = partial(_emission.emission_single_stream, Gauss_quad=2)
    jaxpr = jax.make_jaxpr(f)(T, dz, wl, kappa)
    assert jaxpr is not None
    assert len(jaxpr.jaxpr.eqns) > 0


def test_make_jaxpr_jax_transmission_callback():
    """make_jaxpr on the TRIDENT pure_callback wrapper succeeds.

    The callback itself is opaque to ``jax.grad`` (documented in
    MISMATCHES.md → "v1-C TRIDENT JIT boundary uses jax.pure_callback")
    but ``make_jaxpr`` and ``jax.jit`` go through.
    """
    rng = np.random.default_rng(3)
    N_layers, N_wl = 6, 8
    P = jnp.logspace(-6, 2, N_layers)
    r = jnp.linspace(7.0e7, 7.5e7, N_layers)
    r_up = r + 5.0e4
    r_low = r - 5.0e4
    dr = r_up - r_low
    wl = jnp.linspace(1.0, 10.0, N_wl)
    kappa_clear = jnp.asarray(rng.uniform(1e-9, 1e-7, size=(N_layers, 1, 1, N_wl)))
    kappa_cloud = jnp.zeros_like(kappa_clear)
    phi_edge = jnp.array([-jnp.pi, jnp.pi])
    theta_edge = jnp.array([-jnp.pi / 2, jnp.pi / 2])

    f = lambda P_, kc, kd: _jax_transmission.TRIDENT_callback(  # noqa: E731
        P_,
        r,
        r_up,
        r_low,
        dr,
        wl,
        kc,
        kd,
        0,
        0,
        0.0,
        0.0,
        1.0e9,
        0.0,
        0.0,
        0.0,
        phi_edge,
        theta_edge,
    )
    jaxpr = jax.make_jaxpr(f)(P, kappa_clear, kappa_cloud)
    assert jaxpr is not None


# ---------------------------------------------------------------------------
# Source-grep gate sanity-check: the script runs and exits 0.
# ---------------------------------------------------------------------------
def test_source_grep_gate_passes():
    """The source-grep CI gate exits 0 on the current tree."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    script = repo / "scripts" / "source_grep_gate.py"
    out = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, (
        f"source_grep_gate exited {out.returncode}\n"
        f"stdout:\n{out.stdout}\nstderr:\n{out.stderr}"
    )
    assert "Source-grep gate OK" in out.stdout
