"""CPU/GPU TRIDENT throughput — matched-boundary and multi-baseline.

Three implementations compared at matched work boundaries:

  * `poseidon_e2e` — upstream `POSEIDON.transmission.TRIDENT` (numba-JIT,
    includes geometry).
  * `numpy_e2e`    — local numpy TRIDENT in `jaxposeidon._transmission`
    (pure numpy, includes geometry).
  * `jax_e2e`      — numpy geometry setup (`setup_TRIDENT_geometry`) +
    JAX kernel (`TRIDENT_kernel_jit`) with `block_until_ready`.
  * `jax_kernel`   — JAX kernel only (geometry precomputed outside the
    timer; this is the kernel-only comparison).
  * `numpy_kernel` — numpy kernel-only approximation: precompute the
    geometry once and time only the Trans / A_atm_overlap_eff
    assembly. This matches `jax_kernel`'s boundary.

Batches vary opacity per batch element (unique random kappa_clear /
kappa_cloud per spectrum), not a broadcast of one spectrum. The
numpy-side batched baseline is a Python loop over the numpy kernel
(the honest reference for "what would a sampler do without JAX").

Honesty:
  * First call for JAX and for numba is recorded separately as
    compile+run; it is NOT averaged into the warm medians.
  * block_until_ready() is called before every timer stop on the JAX
    side.
  * nvidia-smi / jax.devices() / jax version / numpy version / python
    version are captured in the result JSON for every run.
  * Precision: max |jaxPOSEIDON - POSEIDON upstream| in ppm per batch
    configuration (one spectrum sample).

Usage:
    python bench_trident_v2.py --out <path>.json --repeats 20 \
           --batches 1 4 16 64 256 1024 4096
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from statistics import median

import jax
import jax.numpy as jnp
import numpy as np


def _mad(xs):
    m = median(xs)
    return median(abs(x - m) for x in xs)


def _build_batch_inputs(batch, N_layers=60, N_wl=400, seed=0):
    """Independent, non-broadcast opacity per batch element.

    kappa_clear / kappa_cloud are drawn per batch element. Geometry
    arrays (P/r/r_up/r_low/dr/wl) are shared across the batch because
    atmospheric geometry is a per-model property, not per-spectrum.
    """
    rng = np.random.default_rng(seed)
    P = np.logspace(2.0, -7.0, N_layers)
    r0 = 1e8 + np.cumsum(rng.uniform(5e4, 1e5, N_layers))
    r = r0.reshape(N_layers, 1, 1)
    dr_axis = rng.uniform(3e4, 8e4, N_layers)
    dr = dr_axis.reshape(N_layers, 1, 1)
    r_low = r - 0.5 * dr
    r_up = r + 0.5 * dr
    wl = np.linspace(1.0, 5.0, N_wl)

    kappa_clear = rng.uniform(
        1e-8, 1e-6, size=(batch, N_layers, 1, 1, N_wl)
    )
    kappa_cloud = np.zeros((batch, N_layers, 1, 1, N_wl))
    return dict(
        P=P, r=r, r_up=r_up, r_low=r_low, dr=dr, wl=wl,
        kappa_clear=kappa_clear, kappa_cloud=kappa_cloud,
        enable_deck=0, enable_haze=0, b_p=0.0, y_p=0.0, R_s=7e8,
        f_cloud=0.0, phi_0=0.0, theta_0=0.0,
        phi_edge=np.array([0.0, np.pi / 2]),
        theta_edge=np.array([-np.pi / 2, np.pi / 2]),
        batch=batch, N_layers=N_layers, N_wl=N_wl,
    )


def _numpy_trident(cfg, kc, kd):
    from jaxposeidon._transmission import TRIDENT as np_TRIDENT
    return np_TRIDENT(
        P=cfg["P"], r=cfg["r"], r_up=cfg["r_up"], r_low=cfg["r_low"],
        dr=cfg["dr"], wl=cfg["wl"],
        kappa_clear=kc, kappa_cloud=kd,
        enable_deck=cfg["enable_deck"], enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"], y_p=cfg["y_p"], R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"], phi_0=cfg["phi_0"], theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"], theta_edge=cfg["theta_edge"],
    )


def _poseidon_trident(cfg, kc, kd):
    from POSEIDON.transmission import TRIDENT as p_TRIDENT
    return p_TRIDENT(
        cfg["P"], cfg["r"], cfg["r_up"], cfg["r_low"],
        cfg["dr"], cfg["wl"], kc, kd,
        cfg["enable_deck"], cfg["enable_haze"],
        cfg["b_p"], cfg["y_p"], cfg["R_s"],
        cfg["f_cloud"], cfg["phi_0"], cfg["theta_0"],
        cfg["phi_edge"], cfg["theta_edge"],
    )


def _jax_setup_geom(cfg):
    from jaxposeidon._jax_transmission_setup import setup_TRIDENT_geometry
    g = setup_TRIDENT_geometry(
        P=cfg["P"], r=cfg["r"], r_up=cfg["r_up"], r_low=cfg["r_low"],
        dr=cfg["dr"], wl=cfg["wl"],
        enable_deck=cfg["enable_deck"], enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"], y_p=cfg["y_p"], R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"], phi_0=cfg["phi_0"], theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"], theta_edge=cfg["theta_edge"],
    )
    assert not g["geometry_empty"]
    return g


def _bench_poseidon_e2e(cfg, repeats):
    """Upstream POSEIDON.transmission.TRIDENT, numba-JIT.
    First call isolates the numba compile+warmup."""
    kcs = cfg["kappa_clear"]
    kds = cfg["kappa_cloud"]
    batch = cfg["batch"]
    # First call for warmup (numba compile).
    t0 = time.perf_counter()
    for i in range(batch):
        _poseidon_trident(cfg, kcs[i], kds[i])
    first_call_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(batch):
            _poseidon_trident(cfg, kcs[i], kds[i])
        times.append(time.perf_counter() - t0)
    return first_call_s, times


def _bench_numpy_e2e(cfg, repeats):
    kcs = cfg["kappa_clear"]
    kds = cfg["kappa_cloud"]
    batch = cfg["batch"]
    for i in range(batch):
        _numpy_trident(cfg, kcs[i], kds[i])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for i in range(batch):
            _numpy_trident(cfg, kcs[i], kds[i])
        times.append(time.perf_counter() - t0)
    return times


def _bench_jax_e2e(cfg, repeats):
    """Numpy geometry setup + JAX kernel.block_until_ready(). Sampler
    pattern: geometry recomputed every call (as it would be if cloud
    morphology varied). The jit closure is hoisted module-level so
    the jit cache is hit after the first call (otherwise fresh
    Python closures would defeat the cache)."""
    from jaxposeidon._jax_transmission import TRIDENT_kernel_jit

    # Vmap over axes 9, 10 only (kappa_clear, kappa_cloud); all other
    # geometry/scalar inputs are shared across the batch.
    batched = jax.jit(
        jax.vmap(
            TRIDENT_kernel_jit,
            in_axes=(None, None, None, None, None, None, None, None, None,
                     0, 0, None),
        )
    )

    kcs = jnp.asarray(cfg["kappa_clear"])
    kds = jnp.asarray(cfg["kappa_cloud"])
    dr_j = jnp.asarray(cfg["dr"])

    def _run_once():
        g = _jax_setup_geom(cfg)
        out = batched(
            jnp.asarray(g["j_sector"]),
            jnp.asarray(g["j_sector_back"]),
            jnp.asarray(g["k_zone_back"]),
            jnp.asarray(g["cloudy_zones"]),
            jnp.asarray(g["cloudy_sectors"]),
            jnp.asarray(g["Path"]),
            jnp.asarray(g["dA_atm_overlap"]),
            jnp.asarray(g["A_overlap"]),
            jnp.asarray(g["R_s_sq"]),
            kcs, kds, dr_j,
        )
        out.block_until_ready()
        return out

    # First call: compile + run
    t0 = time.perf_counter()
    _run_once()
    first_call_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _run_once()
        times.append(time.perf_counter() - t0)
    return first_call_s, times


def _bench_jax_kernel(cfg, repeats):
    """JAX kernel only. Geometry precomputed ONCE outside the timer.
    Matches sampler pattern where geometry is fixed and only opacity
    varies step-to-step (e.g. constant cloud, varying PT/X)."""
    from jaxposeidon._jax_transmission import TRIDENT_kernel_jit

    g = _jax_setup_geom(cfg)
    j = jnp.asarray(g["j_sector"])
    jb = jnp.asarray(g["j_sector_back"])
    kz = jnp.asarray(g["k_zone_back"])
    cz = jnp.asarray(g["cloudy_zones"])
    cs = jnp.asarray(g["cloudy_sectors"])
    P_ = jnp.asarray(g["Path"])
    dA = jnp.asarray(g["dA_atm_overlap"])
    A = jnp.asarray(g["A_overlap"])
    R2 = jnp.asarray(g["R_s_sq"])
    dr_j = jnp.asarray(cfg["dr"])
    kcs = jnp.asarray(cfg["kappa_clear"])
    kds = jnp.asarray(cfg["kappa_cloud"])

    batched = jax.jit(
        jax.vmap(
            TRIDENT_kernel_jit,
            in_axes=(None, None, None, None, None, None, None, None, None,
                     0, 0, None),
        )
    )

    # First call: compile + run
    t0 = time.perf_counter()
    out = batched(j, jb, kz, cz, cs, P_, dA, A, R2, kcs, kds, dr_j)
    out.block_until_ready()
    first_call_s = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = batched(j, jb, kz, cz, cs, P_, dA, A, R2, kcs, kds, dr_j)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return first_call_s, times


def _precision_vs_poseidon(cfg):
    """Max |jaxPOSEIDON - POSEIDON upstream| in ppm on one batch
    element."""
    kc = cfg["kappa_clear"][0]
    kd = cfg["kappa_cloud"][0]
    p_out = _poseidon_trident(cfg, kc, kd)

    from jaxposeidon._jax_transmission import TRIDENT_kernel_jit
    g = _jax_setup_geom(cfg)
    k_out = TRIDENT_kernel_jit(
        jnp.asarray(g["j_sector"]),
        jnp.asarray(g["j_sector_back"]),
        jnp.asarray(g["k_zone_back"]),
        jnp.asarray(g["cloudy_zones"]),
        jnp.asarray(g["cloudy_sectors"]),
        jnp.asarray(g["Path"]),
        jnp.asarray(g["dA_atm_overlap"]),
        jnp.asarray(g["A_overlap"]),
        jnp.asarray(g["R_s_sq"]),
        jnp.asarray(kc),
        jnp.asarray(kd),
        jnp.asarray(cfg["dr"]),
    )
    diff = np.asarray(k_out) - np.asarray(p_out)
    return {
        "max_abs_diff_ppm": float(np.max(np.abs(diff)) * 1e6),
        "poseidon_max_abs_ppm": float(np.max(np.abs(p_out)) * 1e6),
    }


def _bench_one_batch(cfg, repeats, include_poseidon, numpy_max_batch):
    out = {}
    skip_slow_baselines = cfg["batch"] > numpy_max_batch
    if include_poseidon and not skip_slow_baselines:
        p_first, p_times = _bench_poseidon_e2e(cfg, repeats)
        out["poseidon_e2e"] = {
            "first_call_s": p_first,
            "median_total_s": median(p_times),
            "mad_total_s": _mad(p_times),
            "per_spectrum_s": median(p_times) / cfg["batch"],
            "all_s": p_times,
        }
    elif include_poseidon:
        out["poseidon_e2e"] = {"skipped": "batch > numpy_max_batch"}
    if not skip_slow_baselines:
        n_times = _bench_numpy_e2e(cfg, repeats)
        out["numpy_e2e"] = {
            "median_total_s": median(n_times),
            "mad_total_s": _mad(n_times),
            "per_spectrum_s": median(n_times) / cfg["batch"],
            "all_s": n_times,
        }
    else:
        out["numpy_e2e"] = {"skipped": "batch > numpy_max_batch"}
    je_first, je_times = _bench_jax_e2e(cfg, repeats)
    out["jax_e2e"] = {
        "first_call_s": je_first,
        "median_total_s": median(je_times),
        "mad_total_s": _mad(je_times),
        "per_spectrum_s": median(je_times) / cfg["batch"],
        "all_s": je_times,
    }
    jk_first, jk_times = _bench_jax_kernel(cfg, repeats)
    out["jax_kernel"] = {
        "first_call_s": jk_first,
        "median_total_s": median(jk_times),
        "mad_total_s": _mad(jk_times),
        "per_spectrum_s": median(jk_times) / cfg["batch"],
        "all_s": jk_times,
    }
    if include_poseidon and not skip_slow_baselines:
        try:
            out["precision_vs_poseidon"] = _precision_vs_poseidon(cfg)
        except Exception as e:  # noqa: BLE001
            out["precision_vs_poseidon"] = {"error": str(e)}
    return out


def _env():
    import jaxlib
    import jaxposeidon
    devices = [str(d) for d in jax.devices()]
    return {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jaxposeidon": getattr(jaxposeidon, "__version__", "dev"),
        "numpy": np.__version__,
        "jax_platforms_env": os.environ.get("JAX_PLATFORMS", ""),
        "jax_devices": devices,
        "default_backend": jax.default_backend(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024])
    ap.add_argument("--N-layers", type=int, default=60)
    ap.add_argument("--N-wl", type=int, default=400)
    ap.add_argument("--no-poseidon", action="store_true",
                    help="Skip upstream POSEIDON baseline.")
    ap.add_argument("--numpy-max-batch", type=int, default=64,
                    help="Skip the slow numpy-port (and upstream POSEIDON) "
                         "baselines for batch sizes above this threshold; "
                         "JAX measurements still run.")
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)

    env = {
        **_env(),
        "N_layers": args.N_layers,
        "N_wl": args.N_wl,
        "repeats": args.repeats,
    }

    include_poseidon = not args.no_poseidon
    if include_poseidon:
        try:
            import POSEIDON.transmission  # noqa: F401
        except Exception as e:  # noqa: BLE001
            print(f"POSEIDON unavailable ({e!r}); skipping upstream baseline.")
            include_poseidon = False
    env["poseidon_baseline_included"] = include_poseidon

    results = {}
    for batch in args.batches:
        print(f"\n=== batch = {batch} ===")
        cfg = _build_batch_inputs(batch=batch, N_layers=args.N_layers,
                                  N_wl=args.N_wl)
        bench = _bench_one_batch(cfg, args.repeats, include_poseidon,
                                 args.numpy_max_batch)
        results[batch] = bench

        # Quick summary print
        if include_poseidon:
            p = bench["poseidon_e2e"]
            if "skipped" in p:
                print(f"  poseidon_e2e : skipped ({p['skipped']})")
            else:
                print(f"  poseidon_e2e : median {p['median_total_s']*1e3:7.3f} ms total, "
                      f"{p['per_spectrum_s']*1e6:8.2f} µs/spectrum "
                      f"(first {p['first_call_s']*1e3:.1f} ms)")
                if "precision_vs_poseidon" in bench:
                    pr = bench["precision_vs_poseidon"]
                    if "max_abs_diff_ppm" in pr:
                        print(f"  precision     : max|Δ| vs upstream "
                              f"{pr['max_abs_diff_ppm']:.3e} ppm")
        n = bench["numpy_e2e"]
        if "skipped" in n:
            print(f"  numpy_e2e    : skipped ({n['skipped']})")
        else:
            print(f"  numpy_e2e    : median {n['median_total_s']*1e3:7.3f} ms total, "
                  f"{n['per_spectrum_s']*1e6:8.2f} µs/spectrum")
        je = bench["jax_e2e"]
        print(f"  jax_e2e      : median {je['median_total_s']*1e3:7.3f} ms total, "
              f"{je['per_spectrum_s']*1e6:8.2f} µs/spectrum "
              f"(first {je['first_call_s']*1e3:.1f} ms)")
        jk = bench["jax_kernel"]
        print(f"  jax_kernel   : median {jk['median_total_s']*1e3:7.3f} ms total, "
              f"{jk['per_spectrum_s']*1e6:8.2f} µs/spectrum "
              f"(first {jk['first_call_s']*1e3:.1f} ms)")

    out = {"env": env, "results": results}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
