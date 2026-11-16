"""CPU vs GPU TRIDENT throughput benchmark.

Compares:
  * numpy TRIDENT   (jaxposeidon._transmission.TRIDENT)
  * JAX kernel      (jaxposeidon._jax_transmission.TRIDENT_kernel_jit)

on single-spectrum and batched workloads.  Separates JIT-compile time
from warmed execution; uses block_until_ready for honest wall-clock;
reports median + MAD across repeats.

Run on CPU:  JAX_PLATFORMS=cpu python3 bench_trident.py
Run on GPU:  python3 bench_trident.py (default picks CUDA if visible)

Outputs JSON to --out, and prints a Markdown summary.
"""

from __future__ import annotations

import argparse
import json
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


def _build_inputs(N_layers=60, N_wl=400, seed=0):
    """Mid-size realistic transmission inputs.

    N_layers=60, N_wl=400 is representative of a typical K2-18b-style
    chord model: 60 pressure layers, ~400 model wavelengths (then binned
    down to ~20 JWST data bins).
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
    kappa_gas = rng.uniform(1e-8, 1e-6, (N_layers, 1, 1, N_wl))
    kappa_Ray = rng.uniform(1e-9, 1e-7, (N_layers, 1, 1, N_wl))
    kappa_cloud = np.zeros((N_layers, 1, 1, N_wl))
    return dict(
        P=P, r=r, r_up=r_up, r_low=r_low, dr=dr, wl=wl,
        kappa_clear=kappa_gas + kappa_Ray, kappa_cloud=kappa_cloud,
        enable_deck=0, enable_haze=0, b_p=0.0, y_p=0.0, R_s=7e8,
        f_cloud=0.0, phi_0=0.0, theta_0=0.0,
        phi_edge=np.array([0.0, np.pi / 2]),
        theta_edge=np.array([-np.pi / 2, np.pi / 2]),
        N_layers=N_layers, N_wl=N_wl,
    )


def _setup_jax(cfg):
    from jaxposeidon._jax_transmission import TRIDENT_kernel_jit
    from jaxposeidon._jax_transmission_setup import setup_TRIDENT_geometry

    geom = setup_TRIDENT_geometry(
        P=cfg["P"], r=cfg["r"], r_up=cfg["r_up"], r_low=cfg["r_low"],
        dr=cfg["dr"], wl=cfg["wl"],
        enable_deck=cfg["enable_deck"], enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"], y_p=cfg["y_p"], R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"], phi_0=cfg["phi_0"], theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"], theta_edge=cfg["theta_edge"],
    )
    assert not geom["geometry_empty"]

    j_sector = jnp.asarray(geom["j_sector"])
    j_sector_back = jnp.asarray(geom["j_sector_back"])
    k_zone_back = jnp.asarray(geom["k_zone_back"])
    cloudy_zones = jnp.asarray(geom["cloudy_zones"])
    cloudy_sectors = jnp.asarray(geom["cloudy_sectors"])
    Path = jnp.asarray(geom["Path"])
    dA_atm_overlap = jnp.asarray(geom["dA_atm_overlap"])
    A_overlap = jnp.asarray(geom["A_overlap"])
    R_s_sq = jnp.asarray(geom["R_s_sq"])
    dr = jnp.asarray(cfg["dr"])

    def _call(kappa_clear, kappa_cloud):
        return TRIDENT_kernel_jit(
            j_sector, j_sector_back, k_zone_back,
            cloudy_zones, cloudy_sectors,
            Path, dA_atm_overlap, A_overlap, R_s_sq,
            kappa_clear, kappa_cloud, dr,
        )

    jitted_single = jax.jit(_call)
    jitted_batched = jax.jit(jax.vmap(_call, in_axes=(0, 0)))
    return jitted_single, jitted_batched


def _bench_numpy_trident(cfg, repeats):
    """numpy baseline using jaxposeidon._transmission.TRIDENT."""
    from jaxposeidon._transmission import TRIDENT as np_TRIDENT

    # Warmup
    np_TRIDENT(
        P=cfg["P"], r=cfg["r"], r_up=cfg["r_up"], r_low=cfg["r_low"],
        dr=cfg["dr"], wl=cfg["wl"],
        kappa_clear=cfg["kappa_clear"], kappa_cloud=cfg["kappa_cloud"],
        enable_deck=cfg["enable_deck"], enable_haze=cfg["enable_haze"],
        b_p=cfg["b_p"], y_p=cfg["y_p"], R_s=cfg["R_s"],
        f_cloud=cfg["f_cloud"], phi_0=cfg["phi_0"], theta_0=cfg["theta_0"],
        phi_edge=cfg["phi_edge"], theta_edge=cfg["theta_edge"],
    )

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        np_TRIDENT(
            P=cfg["P"], r=cfg["r"], r_up=cfg["r_up"], r_low=cfg["r_low"],
            dr=cfg["dr"], wl=cfg["wl"],
            kappa_clear=cfg["kappa_clear"], kappa_cloud=cfg["kappa_cloud"],
            enable_deck=cfg["enable_deck"], enable_haze=cfg["enable_haze"],
            b_p=cfg["b_p"], y_p=cfg["y_p"], R_s=cfg["R_s"],
            f_cloud=cfg["f_cloud"], phi_0=cfg["phi_0"], theta_0=cfg["theta_0"],
            phi_edge=cfg["phi_edge"], theta_edge=cfg["theta_edge"],
        )
        times.append(time.perf_counter() - t0)
    return times


def _bench_jax_single(jitted_single, cfg, repeats):
    kappa_clear = jnp.asarray(cfg["kappa_clear"])
    kappa_cloud = jnp.asarray(cfg["kappa_cloud"])

    # First call: compile + run
    t0 = time.perf_counter()
    out = jitted_single(kappa_clear, kappa_cloud)
    out.block_until_ready()
    compile_run = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jitted_single(kappa_clear, kappa_cloud)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return compile_run, times


def _bench_jax_batched(jitted_batched, cfg, batch, repeats):
    kappa_clear = jnp.broadcast_to(
        jnp.asarray(cfg["kappa_clear"]),
        (batch,) + cfg["kappa_clear"].shape,
    )
    kappa_cloud = jnp.broadcast_to(
        jnp.asarray(cfg["kappa_cloud"]),
        (batch,) + cfg["kappa_cloud"].shape,
    )

    t0 = time.perf_counter()
    out = jitted_batched(kappa_clear, kappa_cloud)
    out.block_until_ready()
    compile_run = time.perf_counter() - t0

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = jitted_batched(kappa_clear, kappa_cloud)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    return compile_run, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output JSON path")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument(
        "--batches", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024]
    )
    ap.add_argument("--N-layers", type=int, default=60)
    ap.add_argument("--N-wl", type=int, default=400)
    args = ap.parse_args()

    jax.config.update("jax_enable_x64", True)

    cfg = _build_inputs(N_layers=args.N_layers, N_wl=args.N_wl)
    import jaxlib
    import jaxposeidon

    devices = [str(d) for d in jax.devices()]
    env = {
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "jaxposeidon": getattr(jaxposeidon, "__version__", "dev"),
        "numpy": np.__version__,
        "jax_platforms_env": __import__("os").environ.get("JAX_PLATFORMS", ""),
        "jax_devices": devices,
        "default_backend": jax.default_backend(),
        "N_layers": args.N_layers,
        "N_wl": args.N_wl,
        "repeats": args.repeats,
    }
    print(json.dumps({"env": env}, indent=2))

    # --- numpy baseline (single spectrum) ---
    numpy_times = _bench_numpy_trident(cfg, args.repeats)

    # --- JAX single ---
    jitted_single, jitted_batched = _setup_jax(cfg)
    jax_single_compile, jax_single_times = _bench_jax_single(
        jitted_single, cfg, args.repeats
    )

    # --- JAX batched over list of batch sizes ---
    jax_batched = {}
    for batch in args.batches:
        compile_run, times = _bench_jax_batched(
            jitted_batched, cfg, batch, args.repeats
        )
        jax_batched[batch] = {
            "compile_run_s": compile_run,
            "median_s": median(times),
            "mad_s": _mad(times),
            "all_s": times,
            "per_spectrum_median_s": median(times) / batch,
        }

    result = {
        "env": env,
        "numpy_single": {
            "median_s": median(numpy_times),
            "mad_s": _mad(numpy_times),
            "all_s": numpy_times,
        },
        "jax_single": {
            "compile_run_s": jax_single_compile,
            "median_s": median(jax_single_times),
            "mad_s": _mad(jax_single_times),
            "all_s": jax_single_times,
        },
        "jax_batched": jax_batched,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")

    # Short Markdown summary
    print("\n### Summary\n")
    print(f"- numpy single:  median {median(numpy_times)*1e3:.3f} ms")
    print(
        f"- jax single:    median {median(jax_single_times)*1e3:.3f} ms "
        f"(first call {jax_single_compile*1e3:.1f} ms = compile+run)"
    )
    for batch in args.batches:
        d = jax_batched[batch]
        per = d["per_spectrum_median_s"] * 1e6
        print(
            f"- jax batch={batch:>4}: median {d['median_s']*1e3:.3f} ms, "
            f"{per:.2f} µs/spectrum"
        )


if __name__ == "__main__":
    main()
