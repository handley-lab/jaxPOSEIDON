"""Render CPU vs GPU benchmark summary from `bench_turing_*.json`.

Reads the JSON artefacts produced by `scripts/bench_trident.py` and
produces two outputs:

  * `figures/benchmarks.png` — per-spectrum µs vs batch size for
    numpy (CPU), jaxPOSEIDON (CPU), jaxPOSEIDON (CPU/jax), and
    jaxPOSEIDON (GPU).
  * stdout — a Markdown table suitable for pasting into the README
    or an email reply.

Input JSON lives at `benchmarks/bench_turing_cpu.json` and
`benchmarks/bench_turing_gpu.json` by default; override with
`--cpu-json` / `--gpu-json`.
"""

import argparse
import json
import os
from statistics import median

import matplotlib.pyplot as plt


def _load(path):
    with open(path) as f:
        return json.load(f)


def _row(d, batch):
    k = str(batch)
    if k in d["jax_batched"]:
        entry = d["jax_batched"][k]
    else:
        entry = d["jax_batched"][batch]
    return entry


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--cpu-json", default=os.path.join(here, "benchmarks", "bench_turing_cpu.json"))
    ap.add_argument("--gpu-json", default=os.path.join(here, "benchmarks", "bench_turing_gpu.json"))
    ap.add_argument("--out", default=os.path.join(here, "figures", "benchmarks.png"))
    args = ap.parse_args()

    cpu = _load(args.cpu_json)
    gpu = _load(args.gpu_json)

    cpu_batches = sorted(int(k) for k in cpu["jax_batched"])
    gpu_batches = sorted(int(k) for k in gpu["jax_batched"])

    cpu_per = [_row(cpu, b)["per_spectrum_median_s"] * 1e6 for b in cpu_batches]
    gpu_per = [_row(gpu, b)["per_spectrum_median_s"] * 1e6 for b in gpu_batches]

    numpy_single_us = cpu["numpy_single"]["median_s"] * 1e6
    jax_cpu_single_us = cpu["jax_single"]["median_s"] * 1e6
    jax_gpu_single_us = gpu["jax_single"]["median_s"] * 1e6

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhline(
        numpy_single_us,
        color="k", lw=1.0, ls="--",
        label=f"numpy TRIDENT (single) = {numpy_single_us:.0f} µs",
    )
    ax.axhline(
        jax_cpu_single_us,
        color="C0", lw=1.0, ls=":",
        label=f"jax CPU (single, warmed) = {jax_cpu_single_us:.1f} µs",
    )
    ax.axhline(
        jax_gpu_single_us,
        color="C3", lw=1.0, ls=":",
        label=f"jax GPU (single, warmed) = {jax_gpu_single_us:.1f} µs",
    )
    ax.plot(
        cpu_batches, cpu_per, "C0o-",
        label="jax CPU batched (µs/spectrum)",
    )
    ax.plot(
        gpu_batches, gpu_per, "C3s-",
        label="jax GPU batched (µs/spectrum)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("batch size")
    ax.set_ylabel("per-spectrum wall time / µs")
    ax.set_title(
        f"TRIDENT throughput on turing — {gpu['env']['jax_devices'][0]} + "
        f"CPU — N_layers={gpu['env']['N_layers']}, N_wl={gpu['env']['N_wl']}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")

    # Markdown table
    print()
    print("| mode | batch | median total | per-spectrum | speedup vs numpy |")
    print("|---|---:|---:|---:|---:|")
    print(
        f"| numpy single        |    1 | {numpy_single_us:>7.0f} µs | "
        f"{numpy_single_us:>6.0f} µs | 1.0× |"
    )
    print(
        f"| jax CPU single      |    1 | {jax_cpu_single_us:>7.1f} µs | "
        f"{jax_cpu_single_us:>6.1f} µs | {numpy_single_us/jax_cpu_single_us:.0f}× |"
    )
    for b, per in zip(cpu_batches, cpu_per):
        tot = _row(cpu, b)["median_s"] * 1e6
        print(
            f"| jax CPU batch={b:<4} | {b:>4} | {tot:>7.1f} µs | "
            f"{per:>6.1f} µs | {numpy_single_us/per:.0f}× |"
        )
    print(
        f"| jax GPU single      |    1 | {jax_gpu_single_us:>7.1f} µs | "
        f"{jax_gpu_single_us:>6.1f} µs | {numpy_single_us/jax_gpu_single_us:.0f}× |"
    )
    for b, per in zip(gpu_batches, gpu_per):
        tot = _row(gpu, b)["median_s"] * 1e6
        print(
            f"| jax GPU batch={b:<4} | {b:>4} | {tot:>7.1f} µs | "
            f"{per:>6.1f} µs | {numpy_single_us/per:.0f}× |"
        )


if __name__ == "__main__":
    main()
