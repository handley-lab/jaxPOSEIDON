"""v1-C: parity tests for the JAX-traceable TRIDENT transmission RT.

Tolerance default per plan: rtol=1e-13, atol=1e-15 (no-cloud /
canonical Rayleigh) and rtol=1e-11 (Mie). Bit-exact equality is
expected at the TRIDENT_callback boundary because the callback
materialises the numpy `_transmission.TRIDENT` output verbatim;
relaxations are documented in MISMATCHES.md.

Coverage:
- 1D clear (Rayleigh) under jit
- 1D fully-cloudy deck under jit
- cloud_dim=2 patchy cloud under jit
- 3D multi-sector multi-zone configuration under jit
- compute_transmission_spectrum_jit end-to-end (jit of the
  compute_spectrum transmission branch)
- make_jaxpr succeeds on the JIT entry point
- compute_tau_vert_jax pure-jnp kernel parity with the numpy oracle
- trans_from_path_tau_jax pure-jnp kernel parity with numpy
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from jaxposeidon import _transmission  # noqa: E402
from jaxposeidon._compute_spectrum import (  # noqa: E402
    compute_transmission_spectrum_jit,
)
from jaxposeidon._jax_transmission import (  # noqa: E402
    TRIDENT_callback,
    compute_tau_vert_jax,
    trans_from_path_tau_jax,
)


def _canonical_1D_atmosphere(N_layers=50, N_wl=25, seed=0):
    rng = np.random.default_rng(seed)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    R_p = 7.1492e7
    r = np.linspace(R_p, R_p * 1.1, N_layers).reshape(N_layers, 1, 1)
    r_up = np.zeros_like(r)
    r_low = np.zeros_like(r)
    dr = np.zeros_like(r)
    r_up[1:-1, 0, 0] = 0.5 * (r[2:, 0, 0] + r[1:-1, 0, 0])
    r_low[1:-1, 0, 0] = 0.5 * (r[1:-1, 0, 0] + r[:-2, 0, 0])
    dr[1:-1, 0, 0] = 0.5 * (r[2:, 0, 0] - r[:-2, 0, 0])
    r_up[0, 0, 0] = 0.5 * (r[1, 0, 0] + r[0, 0, 0])
    r_up[-1, 0, 0] = r[-1, 0, 0] + 0.5 * (r[-1, 0, 0] - r[-2, 0, 0])
    r_low[0, 0, 0] = r[0, 0, 0] - 0.5 * (r[1, 0, 0] - r[0, 0, 0])
    r_low[-1, 0, 0] = 0.5 * (r[-1, 0, 0] + r[-2, 0, 0])
    dr[0, 0, 0] = r[1, 0, 0] - r[0, 0, 0]
    dr[-1, 0, 0] = r[-1, 0, 0] - r[-2, 0, 0]
    wl = np.linspace(1.0, 5.0, N_wl)
    kappa_clear = rng.uniform(0.0, 1e-5, size=(N_layers, 1, 1, N_wl))
    kappa_cloud = np.zeros_like(kappa_clear)
    return dict(
        P=P,
        r=r,
        r_up=r_up,
        r_low=r_low,
        dr=dr,
        wl=wl,
        kappa_clear=kappa_clear,
        kappa_cloud=kappa_cloud,
    )


def _trident_kwargs(atm, **overrides):
    args = dict(
        b_p=0.0,
        y_p=0.0,
        R_s=6.96e8,
        f_cloud=0.0,
        phi_0=-90.0,
        theta_0=90.0,
        phi_edge=np.array([-np.pi / 2.0, np.pi / 2.0]),
        theta_edge=np.array([-np.pi / 2.0, np.pi / 2.0]),
        enable_deck=0,
        enable_haze=0,
    )
    args.update(overrides)
    args.update(atm)
    return args


def _run_under_jit(atm_kwargs):
    """Invoke TRIDENT_callback under jit and return numpy result."""

    def f(
        P,
        r,
        r_up,
        r_low,
        dr,
        wl,
        kappa_clear,
        kappa_cloud,
        enable_deck,
        enable_haze,
        b_p,
        y_p,
        R_s,
        f_cloud,
        phi_0,
        theta_0,
        phi_edge,
        theta_edge,
    ):
        return TRIDENT_callback(
            P,
            r,
            r_up,
            r_low,
            dr,
            wl,
            kappa_clear,
            kappa_cloud,
            enable_deck,
            enable_haze,
            b_p,
            y_p,
            R_s,
            f_cloud,
            phi_0,
            theta_0,
            phi_edge,
            theta_edge,
        )

    fj = jax.jit(f)
    return np.asarray(
        fj(
            jnp.asarray(atm_kwargs["P"]),
            jnp.asarray(atm_kwargs["r"]),
            jnp.asarray(atm_kwargs["r_up"]),
            jnp.asarray(atm_kwargs["r_low"]),
            jnp.asarray(atm_kwargs["dr"]),
            jnp.asarray(atm_kwargs["wl"]),
            jnp.asarray(atm_kwargs["kappa_clear"]),
            jnp.asarray(atm_kwargs["kappa_cloud"]),
            jnp.asarray(atm_kwargs["enable_deck"], dtype=jnp.int64),
            jnp.asarray(atm_kwargs["enable_haze"], dtype=jnp.int64),
            jnp.asarray(atm_kwargs["b_p"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["y_p"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["R_s"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["f_cloud"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["phi_0"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["theta_0"], dtype=jnp.float64),
            jnp.asarray(atm_kwargs["phi_edge"]),
            jnp.asarray(atm_kwargs["theta_edge"]),
        )
    )


def test_TRIDENT_jit_canonical_rayleigh_1D_matches_numpy():
    atm = _canonical_1D_atmosphere()
    args = _trident_kwargs(atm)
    ours = _run_under_jit(args)
    ref = _transmission.TRIDENT(**args)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_TRIDENT_jit_cloud_deck_1D_matches_numpy():
    atm = _canonical_1D_atmosphere()
    atm["kappa_cloud"] = 1.0e-5 * np.ones_like(atm["kappa_cloud"])
    args = _trident_kwargs(atm, enable_deck=1, f_cloud=1.0, theta_0=-90.0)
    ours = _run_under_jit(args)
    ref = _transmission.TRIDENT(**args)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_TRIDENT_jit_patchy_cloud_dim2_matches_numpy():
    atm = _canonical_1D_atmosphere()
    atm["kappa_cloud"] = 5.0e-6 * np.ones_like(atm["kappa_cloud"])
    args = _trident_kwargs(atm, enable_deck=1, f_cloud=0.4, phi_0=-45.0, theta_0=-90.0)
    ours = _run_under_jit(args)
    ref = _transmission.TRIDENT(**args)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def _canonical_3D_atmosphere(N_layers=30, N_wl=15, N_sectors=2, N_zones=2, seed=1):
    """Build a 3D atmosphere with multiple sectors and zones."""
    rng = np.random.default_rng(seed)
    P = np.logspace(np.log10(100.0), np.log10(1.0e-7), N_layers)
    R_p = 7.1492e7
    r_1d = np.linspace(R_p, R_p * 1.1, N_layers)
    r = np.broadcast_to(r_1d[:, None, None], (N_layers, N_sectors, N_zones)).copy()
    # Slight sector/zone perturbations (still hydrostatically ordered).
    perturb = 1.0 + 1.0e-4 * rng.standard_normal((N_layers, N_sectors, N_zones))
    r = r * perturb
    r_up = np.zeros_like(r)
    r_low = np.zeros_like(r)
    dr = np.zeros_like(r)
    for j in range(N_sectors):
        for k in range(N_zones):
            r_up[1:-1, j, k] = 0.5 * (r[2:, j, k] + r[1:-1, j, k])
            r_low[1:-1, j, k] = 0.5 * (r[1:-1, j, k] + r[:-2, j, k])
            dr[1:-1, j, k] = 0.5 * (r[2:, j, k] - r[:-2, j, k])
            r_up[0, j, k] = 0.5 * (r[1, j, k] + r[0, j, k])
            r_up[-1, j, k] = r[-1, j, k] + 0.5 * (r[-1, j, k] - r[-2, j, k])
            r_low[0, j, k] = r[0, j, k] - 0.5 * (r[1, j, k] - r[0, j, k])
            r_low[-1, j, k] = 0.5 * (r[-1, j, k] + r[-2, j, k])
            dr[0, j, k] = r[1, j, k] - r[0, j, k]
            dr[-1, j, k] = r[-1, j, k] - r[-2, j, k]
    wl = np.linspace(1.0, 5.0, N_wl)
    kappa_clear = rng.uniform(0.0, 1e-5, size=(N_layers, N_sectors, N_zones, N_wl))
    kappa_cloud = np.zeros_like(kappa_clear)
    return dict(
        P=P,
        r=r,
        r_up=r_up,
        r_low=r_low,
        dr=dr,
        wl=wl,
        kappa_clear=kappa_clear,
        kappa_cloud=kappa_cloud,
    )


def test_TRIDENT_jit_3D_multi_sector_zone_matches_numpy():
    atm = _canonical_3D_atmosphere()
    phi_edge = np.linspace(-np.pi / 2.0, np.pi / 2.0, atm["r"].shape[1] + 1)
    theta_edge = np.linspace(-np.pi / 2.0, np.pi / 2.0, atm["r"].shape[2] + 1)
    args = _trident_kwargs(atm, phi_edge=phi_edge, theta_edge=theta_edge)
    ours = _run_under_jit(args)
    ref = _transmission.TRIDENT(**args)
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_compute_transmission_spectrum_jit_matches_numpy():
    atm = _canonical_1D_atmosphere()
    kwargs = _trident_kwargs(
        atm, enable_deck=1, f_cloud=0.4, phi_0=-45.0, theta_0=-90.0
    )
    kwargs["kappa_cloud"] = 5.0e-6 * np.ones_like(atm["kappa_cloud"])
    rng = np.random.default_rng(7)
    kappa_gas = rng.uniform(0, 5e-6, size=atm["kappa_clear"].shape)
    kappa_Ray = atm["kappa_clear"] - kappa_gas  # so kappa_gas + kappa_Ray = kappa_clear

    jit_fn = jax.jit(compute_transmission_spectrum_jit)
    ours = np.asarray(
        jit_fn(
            jnp.asarray(atm["P"]),
            jnp.asarray(atm["r"]),
            jnp.asarray(atm["r_up"]),
            jnp.asarray(atm["r_low"]),
            jnp.asarray(atm["dr"]),
            jnp.asarray(atm["wl"]),
            jnp.asarray(kappa_gas),
            jnp.asarray(kappa_Ray),
            jnp.asarray(kwargs["kappa_cloud"]),
            jnp.int64(kwargs["enable_deck"]),
            jnp.int64(kwargs["enable_haze"]),
            jnp.float64(kwargs["b_p"]),
            jnp.float64(kwargs["y_p"]),
            jnp.float64(kwargs["R_s"]),
            jnp.float64(kwargs["f_cloud"]),
            jnp.float64(kwargs["phi_0"]),
            jnp.float64(kwargs["theta_0"]),
            jnp.asarray(kwargs["phi_edge"]),
            jnp.asarray(kwargs["theta_edge"]),
        )
    )
    ref = _transmission.TRIDENT(
        P=atm["P"],
        r=atm["r"],
        r_up=atm["r_up"],
        r_low=atm["r_low"],
        dr=atm["dr"],
        wl=atm["wl"],
        kappa_clear=(kappa_gas + kappa_Ray),
        kappa_cloud=kwargs["kappa_cloud"],
        enable_deck=kwargs["enable_deck"],
        enable_haze=kwargs["enable_haze"],
        b_p=kwargs["b_p"],
        y_p=kwargs["y_p"],
        R_s=kwargs["R_s"],
        f_cloud=kwargs["f_cloud"],
        phi_0=kwargs["phi_0"],
        theta_0=kwargs["theta_0"],
        phi_edge=kwargs["phi_edge"],
        theta_edge=kwargs["theta_edge"],
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_compute_transmission_spectrum_jit_make_jaxpr_succeeds():
    atm = _canonical_1D_atmosphere(N_layers=10, N_wl=5)
    kwargs = _trident_kwargs(atm)
    kappa_gas = 0.5 * atm["kappa_clear"]
    kappa_Ray = 0.5 * atm["kappa_clear"]
    jaxpr = jax.make_jaxpr(compute_transmission_spectrum_jit)(
        jnp.asarray(atm["P"]),
        jnp.asarray(atm["r"]),
        jnp.asarray(atm["r_up"]),
        jnp.asarray(atm["r_low"]),
        jnp.asarray(atm["dr"]),
        jnp.asarray(atm["wl"]),
        jnp.asarray(kappa_gas),
        jnp.asarray(kappa_Ray),
        jnp.asarray(kwargs["kappa_cloud"]),
        jnp.int64(kwargs["enable_deck"]),
        jnp.int64(kwargs["enable_haze"]),
        jnp.float64(kwargs["b_p"]),
        jnp.float64(kwargs["y_p"]),
        jnp.float64(kwargs["R_s"]),
        jnp.float64(kwargs["f_cloud"]),
        jnp.float64(kwargs["phi_0"]),
        jnp.float64(kwargs["theta_0"]),
        jnp.asarray(kwargs["phi_edge"]),
        jnp.asarray(kwargs["theta_edge"]),
    )
    assert jaxpr is not None
    assert "callback" in str(jaxpr) or "TRIDENT" in str(jaxpr) or len(str(jaxpr)) > 0


def test_compute_tau_vert_jax_matches_numpy_1D_clear():
    """Pure-jnp tau_vert kernel parity (no clouds)."""
    rng = np.random.default_rng(2)
    N_layers, N_wl = 20, 8
    N_sectors_back, N_zones_back = 1, 1
    N_phi, N_zones = 1, 1
    kappa_clear = rng.uniform(
        0, 1e-5, size=(N_layers, N_sectors_back, N_zones_back, N_wl)
    )
    kappa_cloud = rng.uniform(
        0, 1e-6, size=(N_layers, N_sectors_back, N_zones_back, N_wl)
    )
    dr = 1.0e4 * np.ones((N_layers, N_sectors_back, N_zones_back))
    j_sector = np.array([0], dtype=np.int64)
    j_sector_back = np.array([0], dtype=np.int64)
    k_zone_back = np.array([0], dtype=np.int64)
    cloudy_zones = np.array([0], dtype=np.int64)
    cloudy_sectors = np.array([0], dtype=np.int64)
    ref = _transmission.compute_tau_vert(
        N_phi=N_phi,
        N_layers=N_layers,
        N_zones=N_zones,
        N_wl=N_wl,
        j_sector=j_sector,
        j_sector_back=j_sector_back,
        k_zone_back=k_zone_back,
        cloudy_zones=cloudy_zones,
        cloudy_sectors=cloudy_sectors,
        kappa_clear=kappa_clear,
        kappa_cloud=kappa_cloud,
        dr=dr,
    )
    ours = np.asarray(
        compute_tau_vert_jax(
            jnp.asarray(j_sector),
            jnp.asarray(j_sector_back),
            jnp.asarray(k_zone_back),
            jnp.asarray(cloudy_zones),
            jnp.asarray(cloudy_sectors),
            jnp.asarray(kappa_clear),
            jnp.asarray(kappa_cloud),
            jnp.asarray(dr),
        )
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_compute_tau_vert_jax_matches_numpy_cloudy_2zone_2sector():
    rng = np.random.default_rng(3)
    N_layers, N_wl = 20, 8
    N_sectors_back, N_zones_back = 1, 1
    N_phi, N_zones = 2, 2
    kappa_clear = rng.uniform(
        0, 1e-5, size=(N_layers, N_sectors_back, N_zones_back, N_wl)
    )
    kappa_cloud = rng.uniform(
        0, 1e-6, size=(N_layers, N_sectors_back, N_zones_back, N_wl)
    )
    dr = 1.0e4 * np.ones((N_layers, N_sectors_back, N_zones_back))
    j_sector = np.array([0, 1], dtype=np.int64)
    j_sector_back = np.array([0, 0], dtype=np.int64)
    k_zone_back = np.array([0, 0], dtype=np.int64)
    cloudy_zones = np.array([1, 0], dtype=np.int64)
    cloudy_sectors = np.array([1, 0], dtype=np.int64)
    ref = _transmission.compute_tau_vert(
        N_phi=N_phi,
        N_layers=N_layers,
        N_zones=N_zones,
        N_wl=N_wl,
        j_sector=j_sector,
        j_sector_back=j_sector_back,
        k_zone_back=k_zone_back,
        cloudy_zones=cloudy_zones,
        cloudy_sectors=cloudy_sectors,
        kappa_clear=kappa_clear,
        kappa_cloud=kappa_cloud,
        dr=dr,
    )
    ours = np.asarray(
        compute_tau_vert_jax(
            jnp.asarray(j_sector),
            jnp.asarray(j_sector_back),
            jnp.asarray(k_zone_back),
            jnp.asarray(cloudy_zones),
            jnp.asarray(cloudy_sectors),
            jnp.asarray(kappa_clear),
            jnp.asarray(kappa_cloud),
            jnp.asarray(dr),
        )
    )
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)


def test_trans_from_path_tau_jax_matches_numpy():
    """Pure-jnp Beer-Lambert chord kernel parity."""
    rng = np.random.default_rng(4)
    N_b, N_phi, N_zones, N_layers, N_wl = 5, 3, 2, 10, 7
    Path = rng.uniform(0, 1.0, size=(N_b, N_phi, N_zones, N_layers))
    tau_vert = rng.uniform(0, 1e-3, size=(N_layers, N_phi, N_zones, N_wl))
    ref = np.zeros((N_b, N_phi, N_wl))
    for j in range(N_phi):
        ref[:, j, :] = np.exp(
            -np.tensordot(Path[:, j, :, :], tau_vert[:, j, :, :], axes=([2, 1], [0, 1]))
        )
    ours = np.asarray(trans_from_path_tau_jax(jnp.asarray(Path), jnp.asarray(tau_vert)))
    np.testing.assert_allclose(ours, ref, rtol=1e-13, atol=1e-15)
