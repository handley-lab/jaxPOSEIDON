#!/bin/bash
#SBATCH --job-name=jaxp-bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/home/will/work/jaxposeidon-ryan-2026-11-16/slurm-%j.out
#SBATCH --chdir=/home/will/work/jaxposeidon-ryan-2026-11-16

set -euo pipefail
BENCH=/home/will/work/jaxposeidon-ryan-2026-11-16
PY=$BENCH/.venv/bin/python
export PYTHONPATH=$BENCH/worktree
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.1

echo '== host =='; hostname; date -Iseconds
echo '== nvidia-smi =='; nvidia-smi -L
echo '== jax devices =='
$PY -c 'import jax; print(jax.__version__, jax.devices(), jax.default_backend())'

echo
echo '== CPU benchmark (JAX_PLATFORMS=cpu) =='
JAX_PLATFORMS=cpu $PY $BENCH/bench_trident.py \
    --out $BENCH/bench_turing_cpu.json \
    --repeats 20 \
    --batches 1 4 16 64 256 1024

echo
echo '== GPU benchmark =='
$PY $BENCH/bench_trident.py \
    --out $BENCH/bench_turing_gpu.json \
    --repeats 20 \
    --batches 1 4 16 64 256 1024 4096
