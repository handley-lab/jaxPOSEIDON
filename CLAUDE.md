# CLAUDE.md — jaxposeidon coding philosophy

This file codifies the constraints any code in this repository must follow.
It adapts the [jaxwavelets](https://github.com/handley-lab/jaxwavelets)
philosophy (reference-guided translation with adversarial review) to the
POSEIDON-port context.

## 1. POSEIDON is the specification

The numerical oracle is [POSEIDON](https://github.com/MartianColonist/POSEIDON).
Every computation must agree with POSEIDON's output to the
component-specific tolerance documented in the plan
(`~/.claude/plans/let-s-get-going-shimmering-parnas.md`).

- Do **not** re-derive physics from textbooks unless verifying POSEIDON's
  own implementation.
- Do **not** "improve" POSEIDON's algorithm. If POSEIDON does it a certain
  way (including float32 casts, nearest-index lookup, scipy
  `gaussian_filter1d`), this port matches that way.
- If POSEIDON has a bug we can't reproduce, document it as a known mismatch
  in a `MISMATCHES.md` file. Do not patch POSEIDON.

## 2. No defensive programming

This is scientific code, not a web service.

- No input validation past type signatures. If the caller passes garbage,
  the code raises whatever JAX/numpy raises. `let it crash`.
- No `try/except` to "make it robust". If something fails, the failure
  should be visible.
- No "sensible defaults" that silently change behaviour. Match POSEIDON's
  defaults exactly.
- No `assert` for runtime checks beyond what tests need.

## 3. Pure functional

- No module-level globals (other than POSEIDON-extracted constants in
  `_constants.py` and `_species_data.py`).
- No mutable state.
- No singletons / registries / dispatch tables.
- Tests use `pytest` fixtures; data is passed in, not pulled from globals.

## 4. One concept per module

- `_opacity_precompute.py` — opacity preprocessing (log-P interp, wavelength
  sampling/interp, T-interpolation setup).
- `_opacities.py` — runtime extinction (nearest-index lookup,
  cloud/surface thresholds).
- `_atmosphere.py` — T-P profiles, hydrostatic R(P), μ.
- `_chemistry.py` — free chemistry (v0 stub; equilibrium chem deferred).
- `_clouds.py` — MacMad17 deck/haze parameter unpacking.
- `_instruments.py` — convolution, binning.
- `_geometry.py` — angular grids, zone boundaries.
- `_transmission.py` — TRIDENT chord RT.
- `_parameters.py` — v0-branches of POSEIDON parameter/state layer.
- `_data.py` — offsets, error inflation, Gaussian likelihood.
- `_priors.py` — unit-cube prior transform.
- `_constants.py`, `_species_data.py` — build-time-extracted POSEIDON tables.
- `core.py` — public API mirroring POSEIDON's
  (`create_star`, `create_planet`, `define_model`, `read_opacities`,
  `make_atmosphere`, `compute_spectrum`, `load_data`).

Do not cross these concerns. If you need cloud info in a transmission
function, pass it in.

## 5. Tests assert equivalence against POSEIDON

- Every test compares JAX output to POSEIDON output at the same input.
- Tolerances are component-specific (see plan).
- Do not validate against textbook formulas unless POSEIDON also validates
  against them in `tests/test_TRIDENT.py`.

## 6. JAX precision

- `jax.config.update("jax_enable_x64", True)` at module load.
- Match POSEIDON's float32 opacity-table cast where applicable
  (`absorption.py:870`, `:943`).

## 7. Numerics and gradients

- POSEIDON uses nearest-index lookup in `extinction(...)`
  (`absorption.py:1109-1112`, `:1158-1159`). Port this exactly — do not
  smooth it.
- Gradient tests skip points near discontinuities (cloud-deck threshold,
  fine-grid cell boundaries).
- If a smooth surrogate is needed for HMC/VI, place it in a separate
  module (e.g. `_smooth_surrogate.py`); do not pollute the parity path.

## 8. Comments and docstrings

- No comments explaining what well-named code does.
- Document POSEIDON file:line references in docstrings where the port
  mirrors a specific POSEIDON function.
- Do **not** write comments like "matches POSEIDON" or "ported from X" in
  the body — put POSEIDON references in the docstring header.

## 9. Adversarial review gates phase completion

Each implementation phase ends with an `mcp__llm__review` call against
POSEIDON's source. A phase is **not complete** until the review returns
APPROVED.

## 10. License

BSD-3 (matching POSEIDON). Attribute Ryan J. MacDonald 2022 in the
LICENSE file and README.
