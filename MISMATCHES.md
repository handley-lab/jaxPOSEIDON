# jaxposeidon ↔ POSEIDON mismatches

Documented numerical differences between jaxposeidon's v0 port and
POSEIDON reference outputs.

## Open numerical mismatches

### v1-D Toon Thomas tridiagonal: `lax.scan` FP-reorder

POSEIDON's `tri_diag_solve` (`emission.py:534-569`) and `numba_cumsum`
(`emission.py:966-973`) are numba `@jit(nopython=True)` Python `for`
loops over `i` that mutate `AS[i] / DS[i] / XK[i]` in place. The v1-D
JAX port replaces both with `lax.scan` (two-pass backward/forward for
Thomas; cumulative sweep for cumsum). The reduction order matches
POSEIDON's iteration order but the FMA / floating-point reduction is
different, producing ULP-scale residuals (~1e-15 relative, ~1e-19
absolute on standard inputs).

Tests previously using `np.testing.assert_array_equal` were relaxed to
`np.testing.assert_allclose(rtol=1e-13, atol=1e-15)` in
`tests/test_phase_v05_13b_toon.py::test_numba_cumsum_matches_poseidon`
and `::test_tri_diag_solve_matches_poseidon`. Likewise the v0.5
instrument tests (`test_phase8_instruments.py`) were tightened from
`atol=0, rtol=0` to `atol=0, rtol=1e-13` for the same reason — the
underlying `jnp.trapezoid` reduction uses a slightly different
pairwise sum order than `numpy.trapezoid`. The downstream Toon
emission / reflection parity tests run at `rtol=1e-10, atol=1e-8`
per the plan's component-specific tolerance for the source-function
method.
### v1-B FP-reorder tolerance relaxation (rtol=1e-12, atol=1e-7)

The v1-B JAX port of the opacities / atmosphere / chemistry / clouds
hot path replaces `scipy.ndimage.gaussian_filter1d`,
`scipy.interpolate.pchip_interpolate`, and `scipy.special.expn(2, ...)`
with the v1-A JAX primitives (`_jax_filters.gaussian_filter1d_edge`,
`_jax_interpolate.pchip_interpolate`, `_jax_special.expn_2`). The
numerical kernels are mathematically equivalent but the XLA reduction
order differs from scipy / numba, producing ULP-scale residuals:

- `gaussian_filter1d_edge` vs scipy: max rtol ~7e-15 (sigma=3 on a
  100-element float64 array).
- PT-profile `compute_T_Madhu` after Gaussian smoothing: max rtol
  ~2e-16; propagates to `profiles()` outputs.
- `radial_profiles` cumulative trapezoidal integral: max rtol ~2e-16,
  max absolute residual ~1.5e-8 (radius values are ~7e7).
- `compute_T_Guillot` / `compute_T_Guillot_dayside` / `compute_T_Line`:
  max rtol ~2e-13 (powers + `expn_2` reorder).

The v0.5 strict `assert_array_equal` / `rtol=1e-13, atol=0` tests
were relaxed to `rtol=1e-12, atol=1e-7` (the latter covers
radius-scale absolute residuals). The maximum observed residual is
well within both numerical noise and the v0.5 retrieval gradient
sensitivity. No physical observable is affected.

### v1-B `_chemistry.interpolate_log_X_grid` bounds-check guard

POSEIDON `chemistry.py:163-185` raises `Exception` on
out-of-grid-bounds inputs. Under `jax.jit`, the bounds check itself
would force materialisation of the traced query points (since
`np.max` on a traced array isn't defined). The port skips the
Python-side bounds check when called under jit (detected by
`hasattr(log_P, "aval")`), trusting the caller to clip upstream;
the v1-A `regular_grid_interp_linear` primitive itself clips
out-of-range queries to the boundary value (documented as the v1-A
divergence). Out-of-jit calls preserve POSEIDON's exception path
exactly.

### TRIDENT — two paths after v1.0.x real-JAX lift

**State as of v1.0.x real-JAX TRIDENT PR (2026-05-16):** jaxPOSEIDON
now exposes **two** transmission entry points. Honest tradeoff
table:

| Entry point | jit | make_jaxpr | jax.grad through kappa | jax.grad through morphology | Bit-exact vs POSEIDON |
|---|---|---|---|---|---|
| `compute_transmission_spectrum_jit` (legacy `TRIDENT_callback`) | ✓ | ✓ | ✗ (pure_callback no VJP) | ✗ | ✓ |
| `compute_transmission_spectrum_real_jit` (kernel + setup split) | kernel only, NOT public-fn | kernel only | ✓ | ✗ (setup is numpy) | ✓ (parity rtol=1e-13) |

The legacy path is retained for backward compatibility with the v1-C
test suite (which wraps the public function directly in `jax.jit`).
The real-JAX path is the recommended entry point for sampler usage
that needs gradient through atmospheric physics; cloud-morphology
parameters (`f_cloud`, `phi_cloud_0`, `theta_cloud_0`, `enable_deck`,
`enable_haze`, `b_p`, `y_p`, `R_s`) **must be Python scalars** —
they drive output-shape decisions in the geometric setup
(`_jax_transmission_setup.setup_TRIDENT_geometry`, POSEIDON
`transmission.py:289-529, 87-285`) and therefore cannot themselves
be traced.

**Real-JAX path coverage** (`tests/test_v1_real_trident.py`):

- Parametric parity vs POSEIDON numpy at `rtol=1e-13`: 1D cloud-free,
  enable_deck=1, enable_haze=1, nonzero y_p, 2D patchy cloud_dim=2,
  multi-zone (N_zones=2), multi-sector (N_sectors=2).
- `jax.jit(TRIDENT_kernel_jit)` succeeds (kernel takes pre-computed
  geometry tensors as inputs).
- `jax.grad(jax.jit(TRIDENT_kernel_jit))(... , kappa_clear, ...)`
  returns finite, non-zero gradient with finite-difference
  cross-check (`rtol=1e-4`).
- `jax.make_jaxpr(TRIDENT_kernel_jit)(...)` returns a jaxpr that
  does NOT contain `pure_callback`.
- The public `compute_transmission_spectrum_real_jit`'s source is
  inspected to assert it calls `TRIDENT_real_jit`, not
  `TRIDENT_callback` (regression guard).

**Carry-over to v1.x:** Lifting cloud-morphology gradient flow
requires lax-padding the geometric setup (`N_phi_max` /
`N_zones_max` bounds and masks) — a substantial refactor of
`extend_rad_transfer_grids` and `path_distribution_geometric`. Not
in scope for v1.0.x; tracked here as the next gradient-coverage
gate.

### v1-A `_jax_interpolate.regular_grid_interp_linear` boundary handling

The v1-A plan specifies "linear extrapolation off (boundary clip)" for
the JAX RegularGridInterpolator wrapper (plan line: "wrapper around
`jax.scipy.interpolate.RegularGridInterpolator(method='linear')` with
linear extrapolation off (boundary clip)"). scipy's
`RegularGridInterpolator(method='linear', bounds_error=False,
fill_value=None)` extrapolates linearly outside the grid.

This is intentional: the v0.5 callers (`_chemistry.py:121`,
`_clouds.py:162`) construct query points that are clipped upstream to
the grid range, so the extrapolation branch is never exercised. The
JAX wrapper returns the boundary value for out-of-range queries
rather than extrapolating, which is safer under JIT (no spurious
divergence if the upstream clip is removed). Parity tests use only
in-range query points; an explicit clip-vs-extrapolation test
documents the divergence (`test_regular_grid_interp_linear_clips_out_of_range`).

### Phase 0.5.15c LBL handle re-open for multi-(sector, zone)

POSEIDON `absorption.py:1739-1951` opens the molecular and CIA HDF5
opacity handles once before the `(N_sectors, N_zones)` loop, then closes
both inside the loop body (`cia_file.close()` at line 1859 and
`opac_file.close()` at line 1936). For `N_sectors > 1` or `N_zones > 1`
subsequent iterations therefore read from closed `h5py.File` handles
and raise. POSEIDON itself can only execute the `(1, 1)` configuration
in the LBL path.

The port re-opens both handles at the top of each `(j, k)` iteration so
the loop is well-defined for arbitrary `(N_sectors, N_zones)`. On the
`(1, 1)` case the observable output is identical to POSEIDON and the
Phase 0.5.15b bit-equivalence test continues to hold. For
`N_sectors > 1` / `N_zones > 1`, parity is asserted against POSEIDON
run per-`(j, k)` on 1x1 sub-input slices, since the upstream function
cannot complete the multi-dim loop.

### Phase 0.5.14 eddysed: scope and review notes

Phase 0.5.14 (eddysed cloud-model dispatch port) was reviewed
adversarially against POSEIDON `core.py:1685-1700` and
`parameters.py:978-985, 2440-2473`. The first round flagged an
unconditional `kappa_cloud` overwrite (since gated to the
opacity-sampling extinction branch matching POSEIDON's placement)
and a missing `assign_free_params` ordering parity test (since
added, parametrized over `cloud_dim in {1, 2}`). A follow-up round
flagged four pre-existing-in-`main` items that affect all cloud
models (module-level dispatch sets in `_compute_spectrum.py`, a
body-comment file:line reference, ignored
`cloud_properties_contributions` when `kappa_contributions` is
supplied, and an absent `disable_atmosphere` short-circuit in
`assign_free_params`). These are tracked as follow-ups outside
Phase 0.5.14's eddysed scope. The final eddysed-scoped review
returned APPROVED.

### Phase 0.5.12 Mie single-string `aerosol_species`

Same root cause as the FastChem single-string divergence below.
POSEIDON `clouds.py:1711` does
`np.where(aerosol_species == species)[0][0]` after `aerosol_species =
np.array(aerosol_species)` at line 1681; for a single Python string,
this gives a 0-d array and `np.where(...)[0]` raises on numpy >= 2.x.
The port handles the string case explicitly (q=0). Real callers always
pass an iterable; documentation-only divergence.

### Phase 0.5.8 FastChem single-string `chemical_species`

**POSEIDON path crashes in modern numpy.** POSEIDON `chemistry.py:244`
uses `q = np.where(chemical_species == species)[0][0]`. When the
caller passes a Python string for `chemical_species`,
`chemical_species == species` evaluates to a Python `bool`, and
`np.where(bool)[0]` raises `ValueError: Calling nonzero on 0d arrays
is not allowed` on `numpy >= 2.x`.

**Port behaviour.** The port wraps the lookup in
`isinstance(chemical_species, str)` and assigns `q = 0` directly,
which is the documented intent (single-string input means index 0 of
the implicitly singleton grid slice). POSEIDON-parity is asserted
against the single-element-array path instead, since POSEIDON cannot
execute its own string branch.

**Impact.** Real retrievals pass iterable `chemical_species` and never
hit this path in either POSEIDON or jaxposeidon; the divergence is
documentation-only.

### Phase 0.5.17b contribution-kernel index initialization

POSEIDON's `extinction_spectral_contribution` /
`extinction_pressure_contribution` are numba `@jit(nopython=True)`
kernels. When `contribution_species` is not present in either
`chemical_species` or `active_species` (e.g. `'H-'`, which lives in
`bf_species`), the lookup loops at `contributions.py:215-222` /
`:1207-1214` never bind `contribution_molecule_species_index` /
`contribution_molecule_active_index`. Under numba, integer locals
default to `0`; under stock Python they would raise
`UnboundLocalError`. The port initializes both indices to `0` before
the lookup loops, matching numba's observable behaviour. The
`bound_free=True` / `bulk_species=True` / `cloud_contribution=True`
paths guard those references explicitly, so the `0` default is only
read on paths where POSEIDON itself reads it (and there too gets
`0`).

The spectral kernel guards the active-species and Rayleigh loops with
`not bound_free` (see `contributions.py:338`, `:366`); the pressure
kernel does **not** (see `contributions.py:1351`, `:1378`). This
asymmetry is intentional POSEIDON behaviour, mirrored verbatim.

`bound_free` is `True` iff `contribution_species == "H-"` exactly —
not on any of the `bf_species` names like `"H-bf"`. This matches
POSEIDON `contributions.py:232` / `:1224` and is a POSEIDON-inherited
footgun: callers must pass `"H-"` (the species), not the bf_species
label.

## Resolved

- **Phase 4 extinction parametric tolerance**: POSEIDON's numba reduction
  order produces ULP-scale residuals (~4e-25 absolute) on the
  `(haze=1, deck=1)` configuration. The `atol=0, rtol=0` assertion was
  loosened to `atol=1e-22, rtol=1e-13`, well inside FP precision.
- **Phase 2 Madhu signature**: `POSEIDON.atmosphere.profiles` added
  positional kwargs (`T_input`, `X_input`, `P_param_set`,
  `log_P_slope_phot`, `log_P_slope_arr`, `Na_K_fixed_ratio`); the
  rejection test now passes the additional arguments.

- **Phase 0.5.12b review deferred**: OpenAI (`mcp__llm__review` model="openai")
  returned 503 `no_available_accounts` and Gemini returned 429 `RESOURCE_EXHAUSTED`
  at PR-merge time. The Mie cloud runtime port is shipped with parity tests
  passing locally and on CI (Python 3.11 + 3.12); manual review against
  POSEIDON `clouds.py` is a follow-up.

## Phase 0.5.18 final audit — remaining v0.5 gaps (carry to v0.5.x / v0.6)

OpenAI adversarial review of `origin/main` at SHA `bdc6940` (post-#27)
surfaced the following gaps. The corresponding kernels / parameter
plumbing are ported and tested at the unit-function level; the gaps
listed below are at the **public API / dispatch** boundary — i.e. the
kwargs are accepted but the dispatch falls through to
`NotImplementedError` rather than routing to the already-ported kernel.

These are not silent fall-throughs: the user gets a clear
`NotImplementedError` pointing at the follow-up phase. Listed here as
honest scope-deferrals from the v0.5.0 tag.

### Public setup API not yet POSEIDON-free
- `_setup_api.py:205-228` — `define_model`, `read_opacities`,
  `make_atmosphere` still delegate to `POSEIDON.core` rather than the
  native dispatch promised in Phase 0.5.2a's end-state.
- `_loaddata.py:29-31, 54-69` — `init_instrument` / `load_data` ditto.
- Workaround: callers can still use the function — it works — but
  requires POSEIDON installed at runtime.

### Spectrum-type dispatch gaps in `_compute_spectrum` (LIFTED in v1-D)
- `return_albedo=True` — wired: emission / reflection paths return
  `(spectrum, albedo)` when set.
- `spectrum_type="reflection"` — wired via `reflection_Toon` from
  `_emission.py`.
- `thermal_scattering=True` / `reflection=True` in emission — wired
  via `emission_Toon` and reflection_Toon respectively. The
  reflection-on-top-of-emission path adds the reflected-light albedo
  to the emission spectrum (POSEIDON `core.py:1960-1985`).

The lifted dispatcher branches are wired through the JAX-ported
kernels (`emission_Toon` / `reflection_Toon` / `planck_lambda*`) which
are individually jit-parity-tested against POSEIDON oracles
(`test_v1_D_toon_dispatch.py::test_emission_Toon_jit_matches_numpy`,
`::test_reflection_Toon_jit_matches_numpy`, `::test_tri_diag_solve_jit_matches_poseidon`).
The full `compute_spectrum` end-to-end JIT gate against POSEIDON
`core.py:1832-1985` (with non-trivial cloud-scattering inputs:
`w_cloud`, `g_cloud`, `kappa_cloud_seperate` populated by Mie /
eddysed) is the v1-E follow-up — at v1-D the dispatcher routes those
quantities as zero placeholders, which is sufficient for the
cloud-free / single-aerosol envelope but not for the multi-Mie path.
`compute_spectrum` itself is not yet `jax.jit`-able as a top-level
callable because Python dict / string dispatch still dominates the
function body; the JAX-traceability gate is met at the leaf-kernel
level (each ported function is individually jittable and traces under
`make_jaxpr`).

### Setup-api kwarg surface in `create_star` (PARTIALLY LIFTED in v1-D)
- `stellar_contam ∈ {'one_spot', 'two_spots', 'three_spots'}` —
  wired: blackbody `I_phot` / `I_het` computed and threaded through
  `_retrieval.make_loglikelihood` (POSEIDON `stellar.py:797-863`
  pattern). pysynphot / PyMSG stellar grids remain a follow-up.
- non-blackbody `stellar_grid="phoenix"` / `"custom"` still raise.

### Atmosphere-construction gaps
- `_atmosphere.py:1138` — `disable_atmosphere=True` in `profiles(...)`
  raises; bare-rock support lands at the `compute_spectrum` level (PR
  #23) but `profiles` itself does not yet accept the bare-rock path.

### Contribution kernels — partial surface
- `_contributions.py` — `enable_surface=1` and `enable_Mie=1` raise in
  the kernels; the underlying surface (0.5.13d) and Mie (0.5.12b)
  runtime paths are ported but their contribution-channel integration
  is a follow-up.

### Architectural carry-over for v1
- Module-level frozensets in `_compute_spectrum.py:49-51`,
  `_atmosphere.py:35-60, 661-669`, `_retrieval.py:33` are immutable
  dispatch tables and should move to `_constants.py` or to setup-only
  modules per `CLAUDE.md §3`. They function correctly today but are a
  cleanup item before the v1 JAX-trace gate.

## v1.0.0 source-grep grandfather list (v1-E)

The v1-E PR ships the source-grep CI gate
(`scripts/source_grep_gate.py` + `.github/workflows/source-grep-gate.yml`)
and the end-to-end JIT/VMAP/GRAD/make_jaxpr gate
(`tests/test_v1_E_end_to_end.py`). The gate enforces that JAX
**fully-pure** hot-path modules (the 7 modules listed in
"Module classification at v1.0.0") do not perform file I/O —
``v1-grep-skip`` opt-outs are NEVER honored for hard-forbidden
file-I/O patterns in those modules. In the **grandfathered** modules
listed below, hard-forbidden file-I/O patterns are tracked with
explicit per-line ``v1-grep-skip`` opt-outs (with the rationale
inline) rather than gate-failing; the whole module is a v1.0.x
JAX-port follow-up. It additionally tracks but **does not
fail on** residual `import numpy` / `from scipy …` / `import h5py` /
`import sklearn` lines in the modules listed below. Each entry has a
concrete follow-up scope.

Hot-path modules fully JAX-pure at v1.0.0 (7):

- `_data.py`, `_emission.py`, `_opacities.py`, `_parameters.py`,
  `_priors.py`, `_stellar.py`, `_jax_transmission.py`.

Grandfathered (11 modules; v1.0.x follow-ups; line counts as of v1-E
merge):

| Module | Forbidden import count | Follow-up scope |
|---|---|---|
| `_atmosphere.py` | 2 | Port `compute_T_*` setup helpers + `radial_profiles` to jnp; 2D/3D `sector`/`zone` axis vmap; remove `np.zeros` allocations from `profiles(...)` (POSEIDON `atmosphere.py:536-755` non-1D paths). |
| `_chemistry.py` | 1 | Port `interpolate_log_X_grid` Python-shape introspection guards to JAX-safe equivalents; FastChem grid lookup already uses `regular_grid_interp_linear` from v1-A. |
| `_clouds.py` | 1 | Port `unpack_MacMad17_cloud_params` numpy-string-array dispatch to a setup-time builder in `_parameter_setup.py`; runtime kernel (`Mie_cloud`, `interpolate_sigma_Mie_grid`) is already jnp. |
| `_compute_spectrum.py` | 1 | Top-level dispatcher cleanup: remove residual `np.array([0.0])` defaults and `np.empty` allocations in the bare-surface guard; the spectrum-type string dispatch stays in Python (setup-only). |
| `_contributions.py` | 1 | Full JAX port of `extinction_spectral_contribution` + `extinction_pressure_contribution` (`vmap` over wavelength, `lax.fori_loop` over layers, POSEIDON-mirror branch structure preserved). |
| `_high_res.py` | 6 | The largest grandfather surface: `h5py` data loading moves to `_lbl_table_loader.py`; `scipy.optimize.minimize` (`fit_out_transit_spec`) stays setup-only outside jit (POSEIDON convention); `scipy.ndimage.gaussian_filter1d` → `_jax_filters.gaussian_filter1d_edge`; `sklearn.decomposition.TruncatedSVD` → `jax.scipy.linalg.svd` partial-rank wrapper; `scipy.constants` → `_constants.C_M_PER_S`. |
| `_instruments.py` | 1 | Convert `np.asarray` to `jnp.asarray` at entry; integrate setup-time `compute_instrument_indices` from `_instrument_setup.py` so the hot path is allocation-free. |
| `_lbl.py` | 1 | Full JAX port of `compute_kappa_LBL` / `interpolate_sigma_LBL` / `interpolate_cia_LBL` / `T_interpolation_init` to jnp; orchestrator `extinction_LBL` (HDF5 I/O) splits to `_lbl_table_loader.py`. |
| `_opacity_precompute.py` | 1 | Port `closest_index`, `prior_index`, `prior_index_V2`, and the precompute table-fill kernels to jnp; these are the indexing primitives used by both `_opacities.py` and `_lbl.py`. |
| `_retrieval.py` | 1 | The `make_loglikelihood` closure currently uses `np.array([])` for empty parameter blocks; replace with `jnp.array([])` and ensure the closure returns under `jax.jit` for the full top-level `logp(unit_cube)` gate. |
| `_transmission.py` | 1 | The numpy TRIDENT reference; the post-geometry tensor math is already lifted to pure-jnp in `_jax_transmission.TRIDENT_kernel_jit` (v1.0.x real-JAX TRIDENT PR). The remaining numpy is the geometric setup (`extend_rad_transfer_grids`, `path_distribution_geometric`, `delta_ray_geom`, POSEIDON `transmission.py:289-529, 87-285, 633-687`) which produces shape-variable outputs from cloud-morphology scalars. Lifting that into lax-padded form is the v1.x follow-up that enables gradient through geometry. |

### v1-C pure_callback lift — PARTIALLY shipped in v1.0.x real-JAX TRIDENT PR

**Status (2026-05-16)**: A partial lift is shipped. The setup-only
geometric prep stays in numpy (`_jax_transmission_setup.py`); the
post-geometry tensor compute is pure-jnp (`TRIDENT_kernel_jit`); the
public real-JAX entry point is
`compute_transmission_spectrum_real_jit`. `jax.grad` flows through
opacity tensors (`kappa_gas`, `kappa_Ray`, `kappa_cloud`)
end-to-end. **`jax.grad` does NOT flow through geometry-dependent
inputs** (`dr`, `r`, `r_up`, `r_low`, `f_cloud`, `phi_0`, `theta_0`,
`enable_deck`, `enable_haze`, `b_p`, `y_p`, `R_s`) — these flow
through numpy setup which is not traceable.

The legacy `compute_transmission_spectrum_jit` (which routes through
`TRIDENT_callback` / `jax.pure_callback`) is retained for the
v1-C-style "wrap the whole public function in `jax.jit`" usage
pattern (used by `tests/test_v1_C_trident.py`).

Tracked as v1.x follow-up: full lax-padded geometric setup so
gradient flows through morphology/radius inputs too. This requires
fixing `N_phi_max` / `N_zones_max` upper bounds + mask-based
selection in `extend_rad_transfer_grids` and
`path_distribution_geometric`.

### v1-B 2D/3D / Mie deferred sub-items (carried to v1.0.x)

- `_atmosphere.py:profiles(...)` non-1D paths (`N_sectors`/`N_zones` >
  1) use `np.zeros(shape=(N_layers, N_sectors, N_zones))` allocations
  that need to migrate to `jnp.zeros` and vmap over the sector/zone
  axes. The kernels themselves are numerically correct (parity-tested
  in `test_phase_v05_9_2d_3d.py`); the JAX-trace gate is the
  outstanding piece.
- `_clouds.py:Mie_cloud(...)` — aerosol-grid interpolation is jnp at
  the leaf but `np.where`-string-array dispatch in
  `unpack_Mie_cloud_params` needs the `_parameter_setup.py` split.

### v1-E end-to-end JIT/VMAP/GRAD/make_jaxpr gate

The gate at `tests/test_v1_E_end_to_end.py` verifies all five
criteria (jit-parity vs numpy, vmap-of-jit, jit-of-vmap, grad
finiteness, make_jaxpr success) against the *currently jit-traceable
surface* — the leaf kernels that v1-A through v1-D made jit-able
(`planck_lambda_arr`, `emission_single_stream`,
`stellar_contamination_single_spot`, `TRIDENT_callback`). The
top-level `logp(unit_cube)` gate is the v1.0.x follow-up as the
grandfathered modules above are ported and the
`_compute_spectrum.compute_spectrum` outer dispatcher migrates from
Python-string dispatch to a JAX-traceable closure.

The single-stream emission parity check in
`test_emission_single_stream_jit_matches_poseidon` runs at
`rtol=1e-12` rather than the strict `rtol=1e-13` default. This is a
distinct relaxation from the v1-D Toon Thomas tridiagonal entry above
(single-stream does not use Thomas), with the following separate
rationale: the JAX single-stream solver
(`_emission.emission_single_stream`) accumulates per-layer
contributions via a `lax.scan` over the `(N_layers,)` axis, against
POSEIDON's numba Python `for i in range(N_layers)` in-place
accumulation. The reduction order matches but XLA's FMA / pairwise-
sum scheduling differs, producing ULP-scale residuals at the
`(N_layers=20, N_wl=16, Gauss_quad=2)` test configuration:
max observed `rtol ≈ 2.3e-13` (well within `rtol=1e-12` but above the
strict `1e-13` default). This is the same class of XLA-vs-numpy
reduction-order residual as the v1-B FP-reorder entry; the tolerance
is component-specific per the plan's "default rtol=1e-13, relax with
explicit MISMATCHES entry" policy.
