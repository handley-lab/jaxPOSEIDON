"""Real-JAX LBL extinction kernel — parity + grad-flow + jit.

Verifies `_jax_lbl.compute_kappa_LBL_jit` against the numpy
`_lbl.compute_kappa_LBL` over the full (N_layers, N_sectors,
N_zones, N_wl) tensor. The numpy reference is per-(j, k) mutating;
this test loops over (j, k) on the numpy side and compares the
assembled output to the JAX vectorised kernel.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from jaxposeidon import _lbl
from jaxposeidon._jax_lbl import compute_kappa_LBL_jit


def _fixture(
    enable_haze=0,
    enable_deck=0,
    enable_surface=0,
    disable_continuum=False,
    N_sectors=1,
    N_zones=1,
    N_ff=0,
    N_bf=0,
):
    rng = np.random.default_rng(0)
    N_layers = 12
    N_wl = 8
    N_species = 3
    N_species_active = 1
    N_cia_pairs = 2

    P = np.logspace(2.0, -6.0, N_layers)
    wl = np.linspace(1.0, 5.0, N_wl)
    n = rng.uniform(1e15, 1e22, (N_layers, N_sectors, N_zones))

    X = np.zeros((N_species, N_layers, N_sectors, N_zones))
    X[0] = 0.85
    X[1] = 0.149
    X[2] = 1e-3
    X_active = X[2:]

    X_cia = np.zeros((2, N_cia_pairs, N_layers, N_sectors, N_zones))
    X_cia[0, 0] = X[0]
    X_cia[1, 0] = X[0]
    X_cia[0, 1] = X[0]
    X_cia[1, 1] = X[1]

    X_ff = rng.uniform(0.0, 0.5, (2, N_ff, N_layers, N_sectors, N_zones))
    X_bf = rng.uniform(0.0, 1e-4, (N_bf, N_layers, N_sectors, N_zones))

    # sigma_interp / cia_interp / ff_stored are per-layer in LBL (already T-interp'd)
    sigma_interp = rng.uniform(1e-23, 1e-21, (N_species_active, N_layers, N_wl))
    cia_interp = rng.uniform(1e-45, 1e-43, (N_cia_pairs, N_layers, N_wl))
    Rayleigh_stored = rng.uniform(1e-28, 1e-26, (N_species, N_wl))
    ff_stored = rng.uniform(1e-46, 1e-44, (N_ff, N_layers, N_wl))
    bf_stored = rng.uniform(1e-23, 1e-21, (N_bf, N_wl))

    return dict(
        wl_model=wl,
        X=X,
        X_active=X_active,
        X_cia=X_cia,
        X_ff=X_ff,
        X_bf=X_bf,
        n=n,
        P=P,
        a=1.0,
        gamma=-4.0,
        P_cloud=np.array([1e-3]),
        kappa_cloud_0=1e-30,
        N_species=N_species,
        N_species_active=N_species_active,
        N_cia_pairs=N_cia_pairs,
        N_ff_pairs=N_ff,
        N_bf_species=N_bf,
        sigma_interp=sigma_interp,
        cia_interp=cia_interp,
        Rayleigh_stored=Rayleigh_stored,
        ff_stored=ff_stored,
        bf_stored=bf_stored,
        enable_haze=enable_haze,
        enable_deck=enable_deck,
        enable_surface=enable_surface,
        P_surf=1e-30,
        disable_continuum=disable_continuum,
    )


@pytest.mark.parametrize(
    "label,fixture_kwargs",
    [
        ("basic_1D", {}),
        ("multi_sector", {"N_sectors": 2}),
        ("multi_zone", {"N_zones": 2}),
        ("enable_haze", {"enable_haze": 1}),
        ("enable_deck", {"enable_deck": 1}),
        ("enable_surface", {"enable_surface": 1}),
        ("disable_continuum", {"disable_continuum": True}),
        ("with_ff_bf", {"N_ff": 2, "N_bf": 1}),
        ("haze_deck_combo", {"enable_haze": 1, "enable_deck": 1}),
        (
            "all_branches",
            {
                "enable_haze": 1,
                "enable_deck": 1,
                "enable_surface": 1,
                "disable_continuum": True,
                "N_ff": 2,
                "N_bf": 1,
            },
        ),
    ],
)
def test_compute_kappa_LBL_jit_parity_with_numpy(label, fixture_kwargs):
    cfg = _fixture(**fixture_kwargs)
    N_layers = cfg["P"].shape[0]
    N_sectors = cfg["n"].shape[1]
    N_zones = cfg["n"].shape[2]
    N_wl = cfg["wl_model"].shape[0]

    # Run numpy kernel for all (j, k)
    kg_t = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kR_t = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    kc_t = np.zeros((N_layers, N_sectors, N_zones, N_wl))
    for j in range(N_sectors):
        for k in range(N_zones):
            _lbl.compute_kappa_LBL(
                j, k, kappa_gas=kg_t, kappa_Ray=kR_t, kappa_cloud=kc_t,
                **{
                    key: cfg[key]
                    for key in cfg
                    if key not in ("kappa_gas", "kappa_Ray", "kappa_cloud")
                },
            )

    # Run JAX kernel
    kg_o, kR_o, kc_o = compute_kappa_LBL_jit(
        jnp.asarray(cfg["wl_model"]),
        jnp.asarray(cfg["X"]),
        jnp.asarray(cfg["X_active"]),
        jnp.asarray(cfg["X_cia"]),
        jnp.asarray(cfg["X_ff"]),
        jnp.asarray(cfg["X_bf"]),
        jnp.asarray(cfg["n"]),
        jnp.asarray(cfg["P"]),
        cfg["a"],
        cfg["gamma"],
        jnp.asarray(cfg["P_cloud"]),
        cfg["kappa_cloud_0"],
        jnp.asarray(cfg["sigma_interp"]),
        jnp.asarray(cfg["cia_interp"]),
        jnp.asarray(cfg["Rayleigh_stored"]),
        jnp.asarray(cfg["ff_stored"]),
        jnp.asarray(cfg["bf_stored"]),
        cfg["enable_haze"],
        cfg["enable_deck"],
        cfg["enable_surface"],
        cfg["P_surf"],
        cfg["disable_continuum"],
    )

    np.testing.assert_allclose(np.asarray(kg_o), kg_t, rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(kR_o), kR_t, rtol=1e-13, atol=0)
    np.testing.assert_allclose(np.asarray(kc_o), kc_t, rtol=1e-13, atol=0)


def test_compute_kappa_LBL_jit_is_jit_traceable():
    cfg = _fixture()

    @jax.jit
    def _f(X, X_active, X_cia, X_ff, X_bf, sigma_interp, cia_interp,
           Rayleigh_stored, ff_stored, bf_stored, n):
        return compute_kappa_LBL_jit(
            jnp.asarray(cfg["wl_model"]),
            X, X_active, X_cia, X_ff, X_bf, n, jnp.asarray(cfg["P"]),
            cfg["a"], cfg["gamma"],
            jnp.asarray(cfg["P_cloud"]), cfg["kappa_cloud_0"],
            sigma_interp, cia_interp, Rayleigh_stored, ff_stored, bf_stored,
            cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
            cfg["P_surf"], cfg["disable_continuum"],
        )

    kg, kR, kc = _f(
        jnp.asarray(cfg["X"]), jnp.asarray(cfg["X_active"]),
        jnp.asarray(cfg["X_cia"]), jnp.asarray(cfg["X_ff"]),
        jnp.asarray(cfg["X_bf"]),
        jnp.asarray(cfg["sigma_interp"]), jnp.asarray(cfg["cia_interp"]),
        jnp.asarray(cfg["Rayleigh_stored"]),
        jnp.asarray(cfg["ff_stored"]), jnp.asarray(cfg["bf_stored"]),
        jnp.asarray(cfg["n"]),
    )
    assert kg.shape == kR.shape == kc.shape
    assert jnp.all(jnp.isfinite(kg))
    assert jnp.all(jnp.isfinite(kR))
    assert jnp.all(jnp.isfinite(kc))


def test_grad_through_jit_compute_kappa_LBL_via_X_active():
    """jax.grad through X_active is finite + nonzero."""
    cfg = _fixture()

    def _loss(X_active):
        kg, _kR, _kc = compute_kappa_LBL_jit(
            jnp.asarray(cfg["wl_model"]),
            jnp.asarray(cfg["X"]), X_active,
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]), jnp.asarray(cfg["X_bf"]),
            jnp.asarray(cfg["n"]), jnp.asarray(cfg["P"]),
            cfg["a"], cfg["gamma"],
            jnp.asarray(cfg["P_cloud"]), cfg["kappa_cloud_0"],
            jnp.asarray(cfg["sigma_interp"]), jnp.asarray(cfg["cia_interp"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["ff_stored"]), jnp.asarray(cfg["bf_stored"]),
            cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
            cfg["P_surf"], cfg["disable_continuum"],
        )
        return jnp.sum(kg)

    g = jax.grad(jax.jit(_loss))(jnp.asarray(cfg["X_active"]))
    assert g.shape == cfg["X_active"].shape
    assert jnp.all(jnp.isfinite(g))
    assert jnp.any(g != 0.0)

    # Finite-difference cross-check
    flat = jnp.asarray(cfg["X_active"]).reshape(-1)
    idx = jnp.argmax(jnp.abs(g))
    h = 1e-4 * jnp.abs(flat[idx]) + 1e-12
    fp = flat.at[idx].add(h).reshape(cfg["X_active"].shape)
    fm = flat.at[idx].add(-h).reshape(cfg["X_active"].shape)
    fd = (_loss(fp) - _loss(fm)) / (2 * h)
    np.testing.assert_allclose(
        float(g.reshape(-1)[idx]), float(fd), rtol=1e-4, atol=1e-30,
    )


def test_grad_through_jit_compute_kappa_LBL_via_haze_amplitude_a():
    """jax.grad through haze amplitude `a` (with enable_haze=1) is
    finite + nonzero + matches finite difference."""
    cfg = _fixture(enable_haze=1)

    def _loss(a_scalar):
        kg, _kR, kc = compute_kappa_LBL_jit(
            jnp.asarray(cfg["wl_model"]),
            jnp.asarray(cfg["X"]), jnp.asarray(cfg["X_active"]),
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]), jnp.asarray(cfg["X_bf"]),
            jnp.asarray(cfg["n"]), jnp.asarray(cfg["P"]),
            a_scalar, cfg["gamma"],
            jnp.asarray(cfg["P_cloud"]), cfg["kappa_cloud_0"],
            jnp.asarray(cfg["sigma_interp"]), jnp.asarray(cfg["cia_interp"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["ff_stored"]), jnp.asarray(cfg["bf_stored"]),
            cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
            cfg["P_surf"], cfg["disable_continuum"],
        )
        return jnp.sum(kc)  # haze contributes to kappa_cloud

    a0 = jnp.float64(1.0)
    g = jax.grad(jax.jit(_loss))(a0)
    assert jnp.isfinite(g)
    assert g != 0.0
    # Finite-difference cross-check
    h = 1e-4
    fd = (_loss(a0 + h) - _loss(a0 - h)) / (2 * h)
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-6, atol=1e-30)


def test_grad_through_jit_compute_kappa_LBL_via_kappa_cloud_0():
    """jax.grad through deck opacity kappa_cloud_0 (with enable_deck=1)
    is finite + nonzero + matches finite difference."""
    cfg = _fixture(enable_deck=1)
    # Pick P_cloud such that a few layers are above (P > P_cloud).
    cfg["P_cloud"] = np.array([1.0])  # mid-range

    def _loss(kappa_cloud_0_scalar):
        kg, _kR, kc = compute_kappa_LBL_jit(
            jnp.asarray(cfg["wl_model"]),
            jnp.asarray(cfg["X"]), jnp.asarray(cfg["X_active"]),
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]), jnp.asarray(cfg["X_bf"]),
            jnp.asarray(cfg["n"]), jnp.asarray(cfg["P"]),
            cfg["a"], cfg["gamma"],
            jnp.asarray(cfg["P_cloud"]), kappa_cloud_0_scalar,
            jnp.asarray(cfg["sigma_interp"]), jnp.asarray(cfg["cia_interp"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["ff_stored"]), jnp.asarray(cfg["bf_stored"]),
            cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
            cfg["P_surf"], cfg["disable_continuum"],
        )
        return jnp.sum(kc)

    k0 = jnp.float64(1e-3)
    g = jax.grad(jax.jit(_loss))(k0)
    assert jnp.isfinite(g)
    assert g != 0.0
    # FD check
    h = 1e-6
    fd = (_loss(k0 + h) - _loss(k0 - h)) / (2 * h)
    np.testing.assert_allclose(float(g), float(fd), rtol=1e-6, atol=1e-30)


def test_make_jaxpr_compute_kappa_LBL_succeeds():
    cfg = _fixture()

    def _f(X_active):
        return compute_kappa_LBL_jit(
            jnp.asarray(cfg["wl_model"]),
            jnp.asarray(cfg["X"]), X_active,
            jnp.asarray(cfg["X_cia"]),
            jnp.asarray(cfg["X_ff"]), jnp.asarray(cfg["X_bf"]),
            jnp.asarray(cfg["n"]), jnp.asarray(cfg["P"]),
            cfg["a"], cfg["gamma"],
            jnp.asarray(cfg["P_cloud"]), cfg["kappa_cloud_0"],
            jnp.asarray(cfg["sigma_interp"]), jnp.asarray(cfg["cia_interp"]),
            jnp.asarray(cfg["Rayleigh_stored"]),
            jnp.asarray(cfg["ff_stored"]), jnp.asarray(cfg["bf_stored"]),
            cfg["enable_haze"], cfg["enable_deck"], cfg["enable_surface"],
            cfg["P_surf"], cfg["disable_continuum"],
        )

    jaxpr = jax.make_jaxpr(_f)(jnp.asarray(cfg["X_active"]))
    assert jaxpr is not None
    assert "pure_callback" not in str(jaxpr)
