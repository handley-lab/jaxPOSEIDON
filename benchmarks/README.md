# Benchmarks

Recorded benchmark files:
- `bench_turing.sbatch`: Slurm GPU submission script.
- `bench_turing_cpu.json`: Turing CPU results.
- `bench_turing_gpu.json`: Turing GPU results.
- `bench_boltzmann_cpu.json`: Boltzmann CPU results.
- `slurm-7467.out`: Slurm job output.

Benchmarks use `N_layers=60`, `N_wl=400` and `repeats=20`.
Each call is synchronised with `block_until_ready`; the first call is recorded separately as compile+run.
The NumPy baseline is `jaxposeidon._transmission.TRIDENT`, which passes comparison
against POSEIDON with `atol=1e-15` and `rtol=1e-13`.

From the repository root, run `sbatch benchmarks/bench_turing.sbatch` for Slurm GPU
or `python scripts/bench_trident.py --out out.json` for local CPU.

`generate_molecular_validation_figure.py` requires POSEIDON installed.
It builds a synthetic opacity HDF5 file in a temporary directory, then compares
`POSEIDON.core.compute_spectrum` with `jaxposeidon.compute_spectrum`.
