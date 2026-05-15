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
- No singletons / registries / mutable dispatch tables.
- **Immutable POSEIDON-mirror dispatch tables ARE allowed in setup-only
  modules** (`_loaddata.py`, `_instrument_setup.py`,
  `_parameter_setup.py`, `_surface_setup.py`,
  `_stellar_grid_loader.py`, `_fastchem_grid_loader.py`,
  `_aerosol_db_loader.py`, `_eddysed_input_loader.py`,
  `_lbl_table_loader.py`, `_output.py`) where they mirror POSEIDON's
  `reference_data/` dispatch **or** POSEIDON's setup-time
  model/parameter dispatch (e.g. `assign_free_params` parameter-name
  lookups). They must be plain constants — frozen dicts/tuples — and
  must not be mutated at runtime. The "no dispatch tables" rule
  continues to apply to JAX hot-path modules.
- Tests use `pytest` fixtures; data is passed in, not pulled from globals.

## 4. One concept per module

- `_opacity_precompute.py` — opacity preprocessing (log-P interp, wavelength
  sampling/interp, T-interpolation setup).
- `_opacities.py` — runtime extinction (nearest-index lookup,
  cloud/surface thresholds).
- `_atmosphere.py` — T-P profiles, hydrostatic R(P), μ.
- `_chemistry.py` — free chemistry hot path plus FastChem
  equilibrium-chemistry interpolation
  (`interpolate_log_X_grid`, ported from POSEIDON
  `chemistry.py:119-271`). The grid loader stays in
  `_fastchem_grid_loader.py` (setup-only file I/O).
- `_clouds.py` — MacMad17 parameter unpacking + Mie
  `interpolate_sigma_Mie_grid`. Iceberg is **dropped by design**
  (POSEIDON open-source does not implement it at the cloned commit);
  eddysed runtime forward-model integration is the 0.5.14 follow-up.
- `_instruments.py` — JAX-pure convolution / binning hot path
  (setup-only `compute_instrument_indices` + photometric dispatch
  table lives in `_instrument_setup.py`).
- `_geometry.py` — angular grids, zone boundaries; 2D/3D in v0.5.
- `_transmission.py` — TRIDENT chord RT (transmission only).
- `_emission.py` — thermal emission + reflection: Planck radiance,
  single-stream emission (with and without surface emissivity),
  photosphere-radius interpolation, and Toon two-stream
  source-function / reflected-light solvers (ports POSEIDON
  `emission.py:30-1609`). No separate `_reflection.py`; reflected-light
  lives here for parity with POSEIDON's layout.
- Surfaces — there is no `_surfaces.py` module. Surface-albedo
  setup, lab-data dispatch and file I/O live in `_surface_setup.py`;
  the spectral effect (renormalisation, multi-component albedo
  blending, bare-surface emission coupling) is applied directly in
  `_compute_spectrum.py` along the emission flow.
- `_stellar.py` — stellar contamination forward model
  (`planck_lambda`, Rackham+17/18 single-spot,
  multi-region general). The pysynphot / PyMSG grid loader lives in
  `_stellar_grid_loader.py` (setup-only).
- `_lbl.py` — line-by-line opacity mode: `extinction_LBL`
  orchestrator + supporting kernels (the HDF5 loader stays in
  `_lbl_table_loader.py`).
- `_high_res.py` — high-resolution-spectroscopy hot path: airtovac /
  vactoair, sysrem, fast_filter, fit_out_transit_spec, RV-range,
  cross_correlate, rotation kernel, remove_outliers (ports POSEIDON
  `high_res.py:14-29, 179-256, 257-285, 319-336, 339-404, 440-450,
  834-859, 862-882`). The retrieval-closure likelihoods
  (`prepare_high_res_data`, `loglikelihood_PCA`, `loglikelihood_sysrem`,
  `loglikelihood_high_res`) are the 0.5.16b2 follow-up.
- `_contributions.py` — per-species spectral and per-layer pressure
  contribution kernels (`extinction_spectral_contribution`,
  `extinction_pressure_contribution`); POSEIDON-mirror branch
  structure preserved for review parity (allow-listed for `SIM109`,
  `SIM114`, `SIM300` in `pyproject.toml`).
- `_h_minus.py` — H- bound-free and free-free opacities.
- `_setup_api.py` — POSEIDON-setup-API mirror so callers can run
  end-to-end without `import POSEIDON` at runtime (`create_star`,
  `create_planet`, `define_model`, `read_opacities`,
  `make_atmosphere`, `wl_grid_constant_R`).
- `_parameters.py` — hot-path split/unpack of already-constructed
  parameter vectors (JAX-pure in v1). String-heavy
  `assign_free_params`, kwarg dispatch, POSEIDON-mirror dispatch
  tables live in `_parameter_setup.py`.
- `_data.py` — offsets, error inflation, Gaussian likelihood.
- `_priors.py` — unit-cube prior transform (uniform / Gaussian /
  sine in v0; CLR + PT_penalty + 2D/3D Δ-mixing-ratio gating in v0.5).
- `_constants.py`, `_species_data.py` — build-time-extracted POSEIDON
  tables.
- `core.py` — public API mirror. v0 re-exports the ported hot-path
  surface; v0.5 adds POSEIDON setup API (`create_star`,
  `create_planet`, `define_model`, `read_opacities`,
  `make_atmosphere`, `wl_grid_constant_R`) so callers can run
  end-to-end without `import POSEIDON` at runtime.
- `_loaddata.py`, `_instrument_setup.py`, `_parameter_setup.py`,
  `_surface_setup.py`, `_stellar_grid_loader.py`,
  `_fastchem_grid_loader.py`, `_aerosol_db_loader.py`,
  `_eddysed_input_loader.py`, `_lbl_table_loader.py`,
  `_output.py` —
  **setup-only modules**: numpy / scipy / h5py / pysynphot / PyMSG /
  file I/O permitted; never called from inside `jit`; allow-listed by
  the v1 source-grep gate.

Do not cross these concerns. If you need cloud info in a transmission
function, pass it in.

## 5. Tests assert equivalence against POSEIDON

- Where a POSEIDON function is directly callable in isolation, every test
  compares jaxposeidon output to POSEIDON's at the same input
  (`POSEIDON.transmission.TRIDENT`, `POSEIDON.core.compute_spectrum`,
  `POSEIDON.instrument.bin_spectrum_to_data`, etc.).
- Where the POSEIDON code under test lives inside a closure that captures
  retrieval state (POSEIDON's `Prior(...)` inside `run_retrieval`,
  POSEIDON's `LogLikelihood(...)` inside `run_retrieval`,
  `init_instrument(...)` indirectly via `reference_data` dispatch), tests
  compare against a **line-for-line replication of the POSEIDON source**
  with explicit `POSEIDON/...:line` references in the helper. Test names
  end in `_replicates_poseidon_formula_*` and the module docstring
  documents why direct invocation is impossible.
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
