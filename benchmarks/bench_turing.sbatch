#!/bin/bash
#SBATCH --job-name=jaxp-bench-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/home/will/work/jaxposeidon-ryan-2026-11-16/slurm-%j.out
#SBATCH --chdir=/home/will/work/jaxposeidon-ryan-2026-11-16

set -euo pipefail
BENCH=/home/will/work/jaxposeidon-ryan-2026-11-16
PY=$BENCH/.venv/bin/python
export PYTHONPATH=$BENCH/worktree:$BENCH/upstream_poseidon:$BENCH/poseidon_stubs
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.2

echo '== host =='; hostname; date -Iseconds
echo '== nvidia-smi =='; nvidia-smi -L
echo '== jax devices =='
$PY -c 'import jax; print(jax.__version__, jax.devices(), jax.default_backend())'
echo '== POSEIDON available =='
$PY -c 'import POSEIDON.transmission; print(POSEIDON.transmission.TRIDENT)'

echo
echo '== CPU benchmark (JAX_PLATFORMS=cpu) =='
JAX_PLATFORMS=cpu $PY $BENCH/bench_trident_v2.py \
    --out $BENCH/bench_turing_v2_cpu.json \
    --repeats 10 \
    --batches 1 4 16 64 256 1024 \
    --numpy-max-batch 64

echo
echo '== GPU benchmark =='
$PY $BENCH/bench_trident_v2.py \
    --out $BENCH/bench_turing_v2_gpu.json \
    --repeats 20 \
    --batches 1 4 16 64 256 1024 4096 \
    --numpy-max-batch 64
