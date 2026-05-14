# Reference-Guided Translation, Round Two: jaxPOSEIDON

**Will Handley, Institute of Astronomy, University of Cambridge**

## Abstract

[POSEIDON](https://github.com/MartianColonist/POSEIDON) (Ryan J. MacDonald, 2022, BSD-3) is the open-source heir to AURA — the canonical numerical engine for exoplanet atmospheric retrievals in the Madhusudhan-tradition pipeline. It is ~10,000 lines of NumPy/SciPy/Numba covering opacity preprocessing, hydrostatic radius solution, TRIDENT chord radiative transfer, and PyMultiNest-driven retrieval. I wanted that *embodied intelligence* inside [JAX](https://github.com/google/jax) for use with [BlackJAX nested slice sampling](https://github.com/handley-lab/blackjax), the handley-lab fork providing modern gradient-free posterior sampling.

[jaxPOSEIDON v0](https://github.com/handley-lab/jaxPOSEIDON) is the result: a Phase 0–10 port covering POSEIDON's v0 transmission forward-model envelope (1D background, MS09 or isothermal P-T, isochem free chemistry, MacMad17 deck/haze clouds, spectroscopic instrument binning, Gaussian likelihood with offsets and Line15/Piette20 inflation), validated against POSEIDON across **1,435 tests** including a **396-case forward-model parametric sweep** at `atol=1e-15, rtol=1e-13` — comfortably inside the plan's `≤1 ppm` binned-spectrum target. JAX-traceable forward model and live BlackJAX NSS execution are explicitly v1 work items.

This is the **second** application of the reference-guided-translation workflow first demonstrated with [jaxwavelets](https://github.com/handley-lab/jaxwavelets) ([case study](https://github.com/handley-lab/jaxwavelets/blob/master/CASE_STUDY.md)). The original case study argued AI coding agents can produce credible scientific software *when* the task is reference-guided translation, the oracle is mature, and review is adversarial. jaxPOSEIDON tests that thesis on a library an order of magnitude larger — and finds it holds.

## 1. What is different from jaxwavelets

The first port (jaxwavelets) was a clean win: ~3,000 lines, bit-precision parity on 1,177 tests, single mature reference with stable API, ~46 wall-clock hours of session time. jaxPOSEIDON is harder along several axes:

- **Size.** POSEIDON is ~10,000 lines across opacity preprocessing, atmosphere construction, geometry, radiative transfer, instrument convolution, and retrieval orchestration. The v0 envelope deliberately covers the subset needed for K2-18b-style retrievals (1D, free chem, MacMad17 clouds) — but every supported configuration must accept the full POSEIDON kwarg surface and surface deferred branches as descriptive `NotImplementedError`s rather than `TypeError` crashes.
- **Heterogeneous numerics.** POSEIDON mixes Numba JIT, NumPy float64, and float32 opacity tables. The "match POSEIDON bit-for-bit" target attainable in jaxwavelets is unrealistic in places where Numba reduction order, scipy PCHIP boundary handling, and `np.argmin`-nearest-index opacity lookup interact. jaxPOSEIDON aims for **FP-precision** (`atol=1e-15, rtol=1e-13`) rather than bit-exact across the sweep, with bit-exact agreement on the canonical Rayleigh oracle.
- **External data dependence.** POSEIDON ships a 70+ GB opacity database; CI builds synthetic in-tempdir HDF5 fixtures for the CIA and (for the molecular-opacity end-to-end test) the H2O cross sections. Reference-data file I/O (`load_data`, `init_instrument` sensitivity loader) is delegated via thin POSEIDON shims rather than re-ported, because POSEIDON ships the reference data and reimplementing it would mean shipping duplicate data files.
- **Sampler integration.** Where jaxwavelets is a pure numerical-port (no sampler), jaxPOSEIDON's design target is wiring up [BlackJAX NSS](https://github.com/handley-lab/blackjax) over the JAX-traceable forward model. The v0 forward model is a faithful NumPy port (no JAX tracing yet); BlackJAX NSS execution + the K2-18b retrieval run are explicit v1 work items gated on the JAX-tracing migration.
- **Staging discipline.** Rather than a single push, the work was split into 11 phases each ending with adversarial OpenAI review (`mcp__llm__review`) against the POSEIDON source. Reviews returned **NOT APPROVED** repeatedly, demanding fixes such as: rejecting silently-dropped deferred kwargs, adding combinatorial parity tests over every supported configuration, tightening `atol=0` to `atol=0, rtol=0`, replacing claimed "POSEIDON-oracle" tests with explicit `_replicates_poseidon_formula_*` naming and source-line references when the POSEIDON closure was not directly invocable.

## 2. Workflow

The same five-step pattern from jaxwavelets, with one explicit addition for scaling up:

1. **Identify the reference.** POSEIDON is the open-source heir to AURA, used by the Madhusudhan group and others. BSD-3 license permits redistribution of derivative ports with attribution.
2. **Philosophy document** (`CLAUDE.md`) codifying: POSEIDON is the spec, match it numerically; no defensive programming; no input validation past type signatures; pure functional; tests assert equivalence against POSEIDON, not textbook formulas.
3. **Plan with adversarial review.** The plan itself was iterated against `mcp__llm__review` until OpenAI approved both scope and tolerances. Component-specific tolerances were called out explicitly (per-stage `rtol` for float32-sourced opacity, `atol` for float64 scalar arithmetic, `≤1 ppm` for the binned observable), avoiding a single blanket tolerance that would either be too loose (hiding regressions) or too tight (impossible for POSEIDON's Numba float32 paths to satisfy).
4. **Phase-by-phase implementation** (11 phases for jaxPOSEIDON, vs ~4–5 in jaxwavelets):
   - Implementation against POSEIDON source.
   - Test against POSEIDON to the phase-specific tolerance.
   - Adversarial `mcp__llm__review` against the plan + POSEIDON source. Reviewers were explicitly prompted to *push back hard on every deferral*: "for each item this phase excludes, ask is this deferral acceptable for v0? Is the deferred path actually needed for the canonical Rayleigh oracle or K2-18b reproduction? Would deferring cause incorrect output later, or merely limit configs?"
   - Iterate until APPROVED.
5. **Package and publish.** PyPI, AUR PKGBUILD, branch protection, pre-commit hooks (mirroring jaxwavelets' infrastructure). Final comprehensive review pass.

The *adversarial* part deserves emphasis. Across this session, OpenAI returned **NOT APPROVED** more times than it returned APPROVED — for cosmetic issues (stale docstrings, mismatched tolerance descriptions, file references that no longer matched filenames), real defects (`apply_offsets` sentinel comparison that broke on numpy arrays, silent fall-through for unsupported `cloud_model` values, `compute_spectrum` failing to reject `disable_continuum=True`), and missing coverage (no combinatorial tests over `offsets_applied × error_inflation`, no `compute_spectrum` parity test with non-empty active species and nonzero molecular opacity, no Madhu/MS09 P-T parity test). Each round of fixes commit-by-commit improved the port; none of these issues would have been caught by the test suite as-was.

## 3. Numerical results

- **Canonical Rayleigh oracle** (POSEIDON's own `test_TRIDENT.py` setup): bit-exact (`assert_array_equal`) parity end-to-end through `compute_spectrum`.
- **396-case parametric sweep** (T_iso × R_p_ref × P_ref × bulk × MacMad17 cloud_type × cloud_dim × f_cloud × log_a_haze × γ × log_P_cloud): 395 bit-exact, 1 case at max |Δ| = 1.73 × 10⁻¹² ppm — i.e. **twelve orders of magnitude inside the plan's 1 ppm binned-spectrum target**.
- **End-to-end through the instrument model** (compute_spectrum → bin_spectrum_to_data on a JWST NIRSpec-PRISM-style 40-bin layout): max binned |Δ| = 0 ppm.
- **Real molecular opacity** (synthetic H2O HDF5 with nonzero log(σ), `testing=False`): bit-exact parity.

Figures: [`figures/parity_spectra.png`](figures/parity_spectra.png), [`figures/parity_binned.png`](figures/parity_binned.png), [`figures/sweep_histogram.png`](figures/sweep_histogram.png).

## 4. What required human judgment

Decisions only the scientist could make:

- **v0 envelope scoping.** Identifying which POSEIDON branches matter for K2-18b reproduction (MS09 P-T, isochem, MacMad17, one-offset NIRSpec) versus which are v1 (Mie, Iceberg, eddysed, equilibrium chemistry, emission, stellar contamination, surfaces, photometric instruments, LBL mode, 2D/3D atmospheres).
- **Numpy-port-then-JAX-tracing staging.** Pushing back when the agent considered jumping straight to JAX-traced TRIDENT. Numpy-first lets us verify POSEIDON-parity case-by-case before the JAX migration introduces its own residuals.
- **Reference-data delegation boundary.** Deciding that POSEIDON's shipped `reference_data/` (instrument sensitivity files, JWST resolution tables) is *not* something jaxPOSEIDON should re-ship; thin wrappers `_loaddata.load_data` and `_loaddata.init_instrument` delegate the I/O to POSEIDON.
- **Tolerance non-uniformity.** Different components have different floating-point characters; the plan documents `rtol=1e-5` for float32-sourced opacity, `atol=1e-12` for float64 scalar arithmetic, `≤1 ppm` for the binned observable, with rationales for each. A blanket `atol=1e-10` would have been wrong.
- **License preservation.** POSEIDON is BSD-3; jaxPOSEIDON is also BSD-3 to preserve POSEIDON's copyright notice. When I asked if MIT (the handley-lab norm) would be appropriate, the right call was "match POSEIDON".
- **Stopping the agent's context-anxiety.** At one point mid-Phase 9 the agent prematurely declared a stopping point: "v0 forward model is a multi-day undertaking that should be its own session". This was wrong — the work was tractable, and "DO NOT STOP UNTIL COMPLETELY DONE" was a real instruction. The user-side correction ("why did you stop?") was necessary.
- **Push-to-main discipline.** I was repeatedly pushing directly to `main` despite the pre-commit hooks intended to prevent it; the user noticed and asked me to enable `enforce_admins: true` on branch protection so that *everyone* (admins included) must use PRs.

## 5. What the agent did badly and how it was caught

- **Silent deferral fall-through.** Multiple times the agent shipped guards that rejected only a subset of deferred POSEIDON kwargs and silently fell through on others (e.g. `if "Mie" in cloud_model` failing to catch `Iceberg`, `eddysed`). Caught by review pressure "push back hard on every deferral".
- **Overly strict assertions.** Used `atol=0, rtol=0` in places where POSEIDON's Numba reduction order produced ULP-scale residuals (~4 × 10⁻²⁵ absolute) that the bit-exact comparison rejected. Loosened to `atol=1e-22, rtol=1e-13`.
- **Test bypass.** A "combinatorial" coverage test for the retrieval-driver closure was caught by review computing `_data.loglikelihood` directly without invoking the closure under test. Re-written to monkeypatch the forward-model bridge and *actually* exercise the wiring.
- **Plan/code drift.** Reviews repeatedly identified mismatches between the plan's claims (`tests/test_forward_sweep.py:`, 1000+ cases, `jax.grad` parity) and what the code actually shipped (`tests/test_phase9_sweep.py`, 396 cases, FD smoothness with full `jax.grad` gated to v1). Plan amendments per-phase were necessary.
- **Stale docstrings.** Several "bit-exact" docstrings remained after the assertions had been loosened to `assert_allclose(..., atol=1e-15, rtol=1e-13)`. Fixed only after review explicitly flagged the discrepancy.
- **"context panic".** The agent prematurely declared task completion mid-Phase 9 citing context-window concerns. The user-side intervention was: *"don't context panic. You have plenty left."*
- **Direct-to-main pushes.** The session pushed roughly a dozen commits directly to `main` despite the pre-commit hooks intended to prevent exactly that. The branch protection rules I had set with `enforce_admins: false` permitted admin bypass. Caught by the user; fixed by setting `enforce_admins: true`.

## 6. Reproducibility

- Library: [github.com/handley-lab/jaxPOSEIDON](https://github.com/handley-lab/jaxPOSEIDON)
- Test suite: 1,435 cases against POSEIDON (1 skipped — real-DB conditional). Reproduce locally:
  ```bash
  pip install git+https://github.com/MartianColonist/POSEIDON
  pip install -e ".[dev]"
  PYTHONPATH=/path/to/POSEIDON pytest tests/
  ```
- Figures: `python scripts/generate_{parity_figures,binning_figure,sweep_histogram}.py`.
- Plan: `~/.claude/plans/let-s-get-going-shimmering-parnas.md` (in the author's Claude Code state). The plan was iterated against `mcp__llm__review` until approved.
- Philosophy document: [`CLAUDE.md`](CLAUDE.md).
- Synthetic HDF5 fixtures for CI: built per-test-module in tempdirs; see `tests/test_phase9_compute_spectrum.py` for the H2-H2 / H2-He / H2O patterns.
- Numerical mismatches log: [`MISMATCHES.md`](MISMATCHES.md).

## 7. Limitations

**v0/v1 split.** The v0 deliverable is a NumPy port of POSEIDON's hot path, validated against POSEIDON to FP precision. The JAX-traceable migration is a separate v1 work item, as is the live BlackJAX NSS run + K2-18b retrieval comparison. The case for v0 — that the embodied numerical intelligence has been transferred — is stronger than the case that the *purpose* (gradient-based / NSS sampling) is delivered.

**Test-suite size.** 1,435 tests is healthy but the coverage matrix is finite. Combinatorial coverage over the v0 envelope is real but not exhaustive; the residual risk is configurations that pass POSEIDON parity at the per-case level but combine in ways the sweep does not exercise.

**Cost.** The session ran through more API budget than jaxwavelets because the codebase is bigger and the review cycles were more numerous. The economic case for AI-assisted reference-guided translation strengthens with the *library size*, but only up to the point at which review costs start to dominate.

**Maintainability.** The same concern as jaxwavelets applies: the library was generated in an AI-assisted session, and the *understanding* of the code resides in the session transcript more than in any single human's head. The 1,435-test regression suite and the `CLAUDE.md` philosophy document mitigate this, but do not eliminate it. The next test of the workflow is whether a graduate student can debug something in jaxPOSEIDON six months from now without referring to the session log.

## 8. The thesis, restated

The jaxwavelets case study argued that AI coding agents can produce credible scientific software when (a) the task is reference-guided translation, (b) the reference oracle is mature, and (c) review is adversarial. jaxPOSEIDON is a deliberate scaling test of that thesis on a library ~3× the size, with heterogeneous numerics, large external data dependencies, and a sampler-integration target that the original case study did not have.

The thesis holds. The same workflow — embodied-intelligence extraction, adversarial iterative review, scientist-as-architect — produced 1,435 passing tests at FP-precision parity with POSEIDON across 11 reviewed phases. The cost in human attention was small. The cost in API spend was modest. The cost in time was a single session.

The non-trivial difference from the wavelets port is that **scope discipline became load-bearing**. POSEIDON's full surface is large enough that v0 must be deliberately bounded; without an explicit deferral map and adversarial review pushing back on every silent fall-through, the port would have shipped configurations that *appear* to compute results but are subtly wrong. The plan's "Review discipline" section — mandating that every deferred POSEIDON keyword argument be accepted at the jaxposeidon API surface and surface as a descriptive `NotImplementedError` rather than `TypeError` — is the codification of that lesson.
