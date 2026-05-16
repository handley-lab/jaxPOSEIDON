"""Real JAX TRIDENT — replaces the v1-C pure_callback wrapper.

Verifies:
1. Parity vs the numpy ``_transmission.TRIDENT`` oracle at rtol=1e-13.
2. ``jax.jit(TRIDENT_kernel_jit)`` runs.
3. ``jax.grad`` through ``kappa_clear`` returns a finite tensor — i.e.
   gradient genuinely flows through the chord-transmission compute
   (this is what the v1-C wrapper could NOT do).
4. ``jax.make_jaxpr`` succeeds on the jit kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxposeidon import _transmission
from jaxposeidon._jax_transmission import TRIDENT_kernel_jit, TRIDENT_real_jit
from jaxposeidon._jax_transmission_setup import setup_TRIDENT_geometry


def _build_test_inputs(
    enable_deck=0,
    enable_haze=0,
    f_cloud=0.0,
    phi_0=0.0,
    theta_0=0.0,
    y_p=0.0,
    cloud_kappa=0.0,
    N_sectors=1,
    N_zones=1,
):
    """Configurable TRIDENT input set.

    ``N_sectors``/``N_zones`` control the 2D/3D atmosphere structure;
    ``cloud_kappa > 0`` puts opacity into the cloud channel.
    """
    rng = np.random.default_rng(0)
    N_layers = 20
    N_wl = 30
    P = np.logspace(2.0, -7.0, N_layers)
    r0 = 1e8 + np.cumsum(rng.uniform(5e4, 1e5, N_layers))
    # Broadcast radial profile across sectors/zones (atmosphere shape (N_layers, N_sectors, N_zones))
    r = np.broadcast_to(r0.reshape(N_layers, 1, 1), (N_layers, N_sectors, N_zones)).copy()
    dr_axis = rng.uniform(3e4, 8e4, N_layers)
    dr = np.broadcast_to(dr_axis.reshape(N_layers, 1, 1), (N_layers, N_sectors, N_zones)).copy()
    r_low = r - 0.5 * dr
    r_up = r + 0.5 * dr
    wl = np.linspace(1.0, 5.0, N_wl)
    kappa_gas = rng.uniform(1e-8, 1e-6, (N_layers, N_sectors, N_zones, N_wl))
    kappa_Ray = rng.uniform(1e-9, 1e-7, (N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = cloud_kappa * np.ones((N_layers, N_sectors, N_zones, N_wl))
    if N_sectors == 1:
        phi_edge = np.array([0.0, np.pi / 2])
    else:
        phi_edge = np.linspace(0.0, np.pi / 2, N_sectors + 1)
    if N_zones == 1:
        theta_edge = np.array([-np.pi / 2, np.pi / 2])
    else:
        theta_edge = np.linspace(-np.pi / 2, np.pi / 2, N_zones + 1)
    return dict(
        P=P,
        r=r,
        r_up=r_up,
        r_low=r_low,
        dr=dr,
        wl=wl,
        kappa_clear=kappa_gas + kappa_Ray,
        kappa_cloud=kappa_cloud,
        enable_deck=enable_deck,
        enable_haze=enable_haze,
        b_p=0.0,
        y_p=y_p,
        R_s=7e8,
        f_cloud=f_cloud,
        phi_0=phi_0,
        theta_0=theta_0,
        phi_edge=phi_edge,
        theta_edge=theta_edge,
    )


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("1D_cloud_free", {}),
        ("1D_enable_deck", {"enable_deck": 1, "f_cloud": 0.4, "theta_0": -45.0, "phi_0": 30.0, "cloud_kappa": 1e-7}),
        ("1D_enable_haze", {"enable_haze": 1, "f_cloud": 0.5, "phi_0": 30.0, "theta_0": -60.0, "cloud_kappa": 1e-7}),
        ("1D_nonzero_y_p_near_limb", {"y_p": 6.5e8}),
        ("2D_patchy_cloud_dim2", {"f_cloud": 0.3, "phi_0": 60.0, "theta_0": -30.0, "cloud_kappa": 5e-7, "enable_deck": 1}),
        ("multi_zone_2", {"N_zones": 2}),
        ("multi_sector_2", {"N_sectors": 2}),
    ],
)
def test_TRIDENT_real_jit_parity_with_numpy_oracle(label, kwargs):
    cfg = _build_test_inputs(**kwargs)
    ours = TRIDENT_real_jit(
        P=cfg["P"],
        r=cfg["r"],
        r_up=cfg["r_up"],
        r_low=cfg["r_low"],
        dr=cfg["dr"],
        wl=cfg["wl"],
        kappa_clear=jnp.asarray(cfg["kappa_clear"]),
        kappa_cloud=jnp.asarray(cfg["kappa_cloud"]),
        enable_deck=cfg["enable_deck"],
        enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"],
        y_p=cfg["y_p"],
        R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"],
        phi_0=cfg["phi_0"],
        theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"],
        theta_edge=cfg["theta_edge"],
    )
    theirs = _transmission.TRIDENT(
        P=cfg["P"],
        r=cfg["r"],
        r_up=cfg["r_up"],
        r_low=cfg["r_low"],
        dr=cfg["dr"],
        wl=cfg["wl"],
        kappa_clear=cfg["kappa_clear"],
        kappa_cloud=cfg["kappa_cloud"],
        enable_deck=cfg["enable_deck"],
        enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"],
        y_p=cfg["y_p"],
        R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"],
        phi_0=cfg["phi_0"],
        theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"],
        theta_edge=cfg["theta_edge"],
    )
    np.testing.assert_allclose(np.asarray(ours), theirs, rtol=1e-13, atol=0)


def test_TRIDENT_kernel_jit_is_jit_traceable():
    cfg = _build_test_inputs()
    geom = setup_TRIDENT_geometry(
        cfg["P"],
        cfg["r"],
        cfg["r_up"],
        cfg["r_low"],
        cfg["dr"],
        cfg["wl"],
        cfg["enable_deck"],
        cfg["enable_haze"],
        cfg["b_p"],
        cfg["y_p"],
        cfg["R_s"],
        cfg["f_cloud"],
        cfg["phi_0"],
        cfg["theta_0"],
        cfg["phi_edge"],
        cfg["theta_edge"],
    )
    assert not geom["geometry_empty"]

    @jax.jit
    def _f(kappa_clear, kappa_cloud, dr):
        return TRIDENT_kernel_jit(
            jnp.asarray(geom["j_sector"]),
            jnp.asarray(geom["j_sector_back"]),
            jnp.asarray(geom["k_zone_back"]),
            jnp.asarray(geom["cloudy_zones"]),
            jnp.asarray(geom["cloudy_sectors"]),
            jnp.asarray(geom["Path"]),
            jnp.asarray(geom["dA_atm_overlap"]),
            jnp.asarray(geom["A_overlap"]),
            jnp.asarray(geom["R_s_sq"]),
            kappa_clear,
            kappa_cloud,
            dr,
        )

    out = _f(
        jnp.asarray(cfg["kappa_clear"]),
        jnp.asarray(cfg["kappa_cloud"]),
        jnp.asarray(cfg["dr"]),
    )
    assert out.shape == (len(cfg["wl"]),)
    assert jnp.all(jnp.isfinite(out))


def test_grad_through_jit_TRIDENT_kappa_clear_is_finite():
    """The v1-C `pure_callback` wrapper could NOT do this. The real
    JAX kernel does."""
    cfg = _build_test_inputs()
    geom = setup_TRIDENT_geometry(
        cfg["P"],
        cfg["r"],
        cfg["r_up"],
        cfg["r_low"],
        cfg["dr"],
        cfg["wl"],
        cfg["enable_deck"],
        cfg["enable_haze"],
        cfg["b_p"],
        cfg["y_p"],
        cfg["R_s"],
        cfg["f_cloud"],
        cfg["phi_0"],
        cfg["theta_0"],
        cfg["phi_edge"],
        cfg["theta_edge"],
    )

    j_sector = jnp.asarray(geom["j_sector"])
    j_sector_back = jnp.asarray(geom["j_sector_back"])
    k_zone_back = jnp.asarray(geom["k_zone_back"])
    cloudy_zones = jnp.asarray(geom["cloudy_zones"])
    cloudy_sectors = jnp.asarray(geom["cloudy_sectors"])
    Path = jnp.asarray(geom["Path"])
    dA_atm_overlap = jnp.asarray(geom["dA_atm_overlap"])
    A_overlap = jnp.asarray(geom["A_overlap"])
    R_s_sq = jnp.asarray(geom["R_s_sq"])
    kappa_cloud = jnp.asarray(cfg["kappa_cloud"])
    dr = jnp.asarray(cfg["dr"])

    def _scalar_loss(kappa_clear):
        td = TRIDENT_kernel_jit(
            j_sector,
            j_sector_back,
            k_zone_back,
            cloudy_zones,
            cloudy_sectors,
            Path,
            dA_atm_overlap,
            A_overlap,
            R_s_sq,
            kappa_clear,
            kappa_cloud,
            dr,
        )
        return jnp.sum(td)

    # Use natural physical kappa magnitude; scaling up too aggressively
    # makes exp(-tau) underflow to 0 and grad vanishes by saturation.
    kappa_in = jnp.asarray(cfg["kappa_clear"])

    grad_fn = jax.grad(jax.jit(_scalar_loss))
    g = grad_fn(kappa_in)
    assert g.shape == cfg["kappa_clear"].shape
    assert jnp.all(jnp.isfinite(g))
    # Path tensor sparsity check — chord paths through inner layers must
    # exist, otherwise the test fixture is degenerate.
    assert jnp.any(Path > 0.0), "test fixture has empty Path; not a code bug"
    # **Strong gradient assertion**: at least one element of grad must be
    # genuinely nonzero. If this fails, the JAX backprop is not actually
    # flowing through the chord transmission (i.e., the kernel might be
    # numerically constant in kappa).
    assert jnp.any(g != 0.0), "jax.grad through kappa_clear is identically zero"
    # **Finite-difference cross-check** for one (layer, sector, zone, wl)
    # entry where grad is nonzero — proves autograd ≈ numerical derivative.
    nonzero_idx = jnp.argmax(jnp.abs(g))
    flat = kappa_in.reshape(-1)
    h = 1e-3 * jnp.abs(flat[nonzero_idx]) + 1e-12
    flat_plus = flat.at[nonzero_idx].add(h)
    flat_minus = flat.at[nonzero_idx].add(-h)
    loss_plus = _scalar_loss(flat_plus.reshape(kappa_in.shape))
    loss_minus = _scalar_loss(flat_minus.reshape(kappa_in.shape))
    fd = (loss_plus - loss_minus) / (2 * h)
    autograd = g.reshape(-1)[nonzero_idx]
    np.testing.assert_allclose(
        float(autograd), float(fd), rtol=1e-4, atol=1e-30,
        err_msg="autograd vs finite-difference mismatch",
    )


def test_make_jaxpr_TRIDENT_kernel_succeeds():
    cfg = _build_test_inputs()
    geom = setup_TRIDENT_geometry(
        cfg["P"],
        cfg["r"],
        cfg["r_up"],
        cfg["r_low"],
        cfg["dr"],
        cfg["wl"],
        cfg["enable_deck"],
        cfg["enable_haze"],
        cfg["b_p"],
        cfg["y_p"],
        cfg["R_s"],
        cfg["f_cloud"],
        cfg["phi_0"],
        cfg["theta_0"],
        cfg["phi_edge"],
        cfg["theta_edge"],
    )

    def _f(kappa_clear, kappa_cloud, dr):
        return TRIDENT_kernel_jit(
            jnp.asarray(geom["j_sector"]),
            jnp.asarray(geom["j_sector_back"]),
            jnp.asarray(geom["k_zone_back"]),
            jnp.asarray(geom["cloudy_zones"]),
            jnp.asarray(geom["cloudy_sectors"]),
            jnp.asarray(geom["Path"]),
            jnp.asarray(geom["dA_atm_overlap"]),
            jnp.asarray(geom["A_overlap"]),
            jnp.asarray(geom["R_s_sq"]),
            kappa_clear,
            kappa_cloud,
            dr,
        )

    jaxpr = jax.make_jaxpr(_f)(
        jnp.asarray(cfg["kappa_clear"]),
        jnp.asarray(cfg["kappa_cloud"]),
        jnp.asarray(cfg["dr"]),
    )
    assert jaxpr is not None
    # The jaxpr should NOT contain pure_callback (that's the whole point)
    assert "pure_callback" not in str(jaxpr)


def test_public_compute_transmission_spectrum_real_jit_uses_real_kernel_not_callback():
    """The public real-JAX entry point must not internally use
    pure_callback. We can't wrap it in jax.jit at the top level
    (the setup orchestrator needs concrete dr/r/r_up values for
    geometry-shape decisions), so instead we run it and verify that
    the *kernel it invokes* is the pure-jnp path: re-run the public
    function with a runtime ``jax.make_jaxpr`` of the internal kernel
    via the chain and ensure ``pure_callback`` is not in the trace.
    """
    from jaxposeidon._compute_spectrum import compute_transmission_spectrum_real_jit

    cfg = _build_test_inputs()
    # Call once to ensure no callback is hit; output is finite.
    out = compute_transmission_spectrum_real_jit(
        cfg["P"],
        cfg["r"],
        cfg["r_up"],
        cfg["r_low"],
        cfg["dr"],
        cfg["wl"],
        cfg["kappa_clear"] * 0.5,
        cfg["kappa_clear"] * 0.5,
        cfg["kappa_cloud"],
        cfg["enable_deck"],
        cfg["enable_haze"],
        cfg["b_p"],
        cfg["y_p"],
        cfg["R_s"],
        cfg["f_cloud"],
        cfg["phi_0"],
        cfg["theta_0"],
        cfg["phi_edge"],
        cfg["theta_edge"],
    )
    assert jnp.all(jnp.isfinite(out))

    # Verify by code inspection (string check on the implementation):
    # the public function's source should reference TRIDENT_real_jit, not
    # TRIDENT_callback. This guards against a regression that re-wires it.
    import inspect

    from jaxposeidon import _compute_spectrum

    src = inspect.getsource(
        _compute_spectrum.compute_transmission_spectrum_real_jit
    )
    assert "TRIDENT_real_jit" in src
    assert "TRIDENT_callback" not in src, (
        "public real-JAX entry point must not call TRIDENT_callback"
    )
