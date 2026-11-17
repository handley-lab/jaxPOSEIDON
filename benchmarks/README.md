# Benchmarks

Matched-boundary TRIDENT throughput for upstream POSEIDON (numba-JIT CPU),
our local numpy port, and jaxPOSEIDON's JAX kernel on CPU and GPU, with
precision vs upstream reported to zero ppm.

Files:
- `bench_turing.sbatch` — Slurm sbatch for the CPU + GPU run.
- `bench_turing_cpu.json` — turing CPU results (JAX_PLATFORMS=cpu).
- `bench_turing_gpu.json` — turing GPU results (CUDA on a single RTX PRO
  6000 Blackwell Max-Q).
- `slurm-7469.out` — full Slurm log for the production run.
- `synthetic_opacity_provenance.json` — grid ranges, random seed, IID
  distribution, atmosphere settings and parity measurements for the
  synthetic H2O opacity used in
  `scripts/generate_synthetic_opacity_parity_figure.py`.

Design:
- Same geometry inputs across all baselines. `poseidon_e2e` and
  `numpy_e2e` include geometry setup; `jax_e2e` runs the numpy
  geometry setup then the JAX kernel with `block_until_ready`;
  `jax_kernel` precomputes geometry once outside the timer.
- Batches use independent `kappa_clear` / `kappa_cloud` per element
  (unique opacity per batch index), not a broadcast-identical
  spectrum.
- First call is recorded separately; the median is taken over
  repeats after warmup.
- Precision vs upstream `POSEIDON.transmission.TRIDENT` is reported
  as max |residual| in ppm on the first batch element.

Run from the repository root:
- `sbatch benchmarks/bench_turing.sbatch` — CPU + GPU on turing.
- `python scripts/bench_trident.py --out run.json --repeats 20 \
     --batches 1 4 16 64 256 1024 --numpy-max-batch 64`

`--numpy-max-batch 64` skips the slow numpy-port and upstream POSEIDON
baselines for batch > 64; JAX measurements still run. The JAX kernel
is unchanged by this flag.

`scripts/generate_synthetic_opacity_parity_figure.py` is a related
script that emits `figures/parity_synthetic_h2o.png` — parity of
`jaxposeidon.compute_spectrum` against upstream POSEIDON on a
synthetic H2O opacity (log sigma ~ U(-23, -21)). It is a data-path
smoke, not a realistic-atmosphere validation.
# Benchmarks

Matched-boundary TRIDENT throughput for upstream POSEIDON (numba-JIT CPU),
our local numpy port, and jaxPOSEIDON's JAX kernel on CPU and GPU, with
precision vs upstream reported to zero ppm.

Files:
- `bench_turing.sbatch` — Slurm sbatch for the CPU + GPU run.
- `bench_turing_cpu.json` — turing CPU results (JAX_PLATFORMS=cpu).
- `bench_turing_gpu.json` — turing GPU results (CUDA on a single RTX PRO
  6000 Blackwell Max-Q).
- `slurm-7469.out` — full Slurm log for the production run.
- `synthetic_opacity_provenance.json` — grid ranges, random seed, IID
  distribution, atmosphere settings and parity measurements for the
  synthetic H2O opacity used in
  `scripts/generate_synthetic_opacity_parity_figure.py`.

Design:
- Same geometry inputs across all baselines. `poseidon_e2e` and
  `numpy_e2e` include geometry setup; `jax_e2e` runs the numpy
  geometry setup then the JAX kernel with `block_until_ready`;
  `jax_kernel` precomputes geometry once outside the timer.
- Batches use independent `kappa_clear` / `kappa_cloud` per element
  (unique opacity per batch index), not a broadcast-identical
  spectrum.
- First call is recorded separately; the median is taken over
  repeats after warmup.
- Precision vs upstream `POSEIDON.transmission.TRIDENT` is reported
  as max |residual| in ppm on the first batch element.

Run from the repository root:
- `sbatch benchmarks/bench_turing.sbatch` — CPU + GPU on turing.
- `python scripts/bench_trident.py --out run.json --repeats 20 \
     --batches 1 4 16 64 256 1024 --numpy-max-batch 64`

`--numpy-max-batch 64` skips the slow numpy-port and upstream POSEIDON
baselines for batch > 64; JAX measurements still run. The JAX kernel
is unchanged by this flag.

`scripts/generate_synthetic_opacity_parity_figure.py` is a related
script that emits `figures/parity_synthetic_h2o.png` — parity of
`jaxposeidon.compute_spectrum` against upstream POSEIDON on a
synthetic H2O opacity (log sigma ~ U(-23, -21)). It is a data-path
smoke, not a realistic-atmosphere validation.

`realistic_tutorial_provenance.json` records the grid ranges, atmosphere settings
and parity measurements for `scripts/generate_realistic_tutorial_parity_figure.py`.
The script reproduces upstream POSEIDON's first transmission tutorial on HAT-P-26b
with real H2O/CH4 line opacity from the v1.3 Zenodo inputs (record 19595136),
yielding `max |residual| = 0` ppm across 21595 wavelength points against upstream
`POSEIDON.core.compute_spectrum`.

The realistic parity test requires `POSEIDON_input_data` to point at an unpacked
`inputs/` directory containing `opacity/Opacity_database_v1.3.hdf5` (66 GB) and
`opacity/Opacity_database_cia.hdf5` (49 MB). Fetch the inputs from Zenodo DOI
[10.5281/zenodo.19595136](https://doi.org/10.5281/zenodo.19595136) (72 GB zip).
Unzip with `UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE`, since the inputs zip hits
unzip's heuristic.

Produced on turing (Slurm `gpu` partition; 1x NVIDIA RTX PRO 6000 Blackwell Max-Q)
with jax 0.12.0.1, jaxlib 0.12.0.1 and Python 3.12.13.
