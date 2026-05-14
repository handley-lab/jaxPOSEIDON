# POSEIDON pinned commit (v0.5 oracle)

v0.5 work is pinned to a specific POSEIDON commit to make
reference-guided translation reproducible. If POSEIDON ships new
features (e.g. an Iceberg implementation) in a later commit, decide
per-feature whether to bump.

| Field | Value |
|---|---|
| Repo | https://github.com/MartianColonist/POSEIDON |
| Commit | `594f6f563269e37dde1571bee44a81c6f33d1f63` |
| Branch | `main` at pin date |
| License | BSD-3-Clause (Ryan J. MacDonald, 2022) |

## Public-API surface (Phase 0.5.1 audit)

297 public callables across 16 modules. The audit table is maintained
in the plan file
(`~/.claude/plans/let-s-get-going-shimmering-parnas.md` — section
"Phase 0.5.1 deliverable: POSEIDON public-API inventory") and below.

### Disposition summary

| Disposition | Count | Notes |
|---|---|---|
| Already ported in v0 | 5 | `compute_spectrum`, `Rayleigh_cross_section`, hot-path subset |
| Port in v0.5 | ~115 | non-v0 PT profiles, FastChem, Mie/eddysed, emission/reflection, stellar, surfaces, LBL, high-res, photometric, setup API |
| Plotting / visualisation | ~30 | `visuals.py` + `clouds.py:plot_*` — out of v0.5 scope (numerics only) |
| Setup-only / I/O | ~40 | `utility.py:write_*`/`read_*`, MultiNest output, MPI shared memory |
| GPU duplicates | ~10 | `extinction_GPU`, `planck_lambda_arr_GPU`, etc. — superseded by v1 JAX backend |
| **Iceberg** | 0 | POSEIDON does **not** implement at this commit; DROPPED from v0.5 |
| **PyMultiNest dispatch** | 0 | intentionally excluded; user's BlackJAX session is the v1+ retrieval driver |
| Legacy / deprecated | ~3 | `precompute_stellar_spectra_OLD`, `radial_profiles_test` |

Full per-callable table: see the v0.5 plan file.

### Module-level summary

| POSEIDON module | Public functions | Phase |
|---|---:|---|
| `core.py` | 12 | 0.5.2a setup API skeleton + extensions per phase |
| `instrument.py` | 10 | 0.5.5 photometric + 0.5.2a init_instrument extract |
| `absorption.py` | 18 | 0.5.4 ff/bf, 0.5.15 LBL, v0 has the rest |
| `atmosphere.py` | 23 | 0.5.6 non-v0 PT, 0.5.7 gradient composition, 0.5.9 2D/3D, 0.5.8 FastChem |
| `geometry.py` | 3 | 0.5.9 (non-1D) |
| `parameters.py` | 7 | 0.5.6/0.5.7/0.5.9/0.5.10 (per branch); setup-heavy moves to `_parameter_setup.py` in 0.5.2a |
| `chemistry.py` | 2 | 0.5.8 FastChem |
| `clouds.py` | 33 numerical + ~31 plotting | 0.5.12 Mie + 0.5.14 eddysed (plotting out of scope) |
| `stellar.py` | 9 | 0.5.11 |
| `surfaces.py` | 3 | 0.5.3 parsing/interp + 0.5.13d spectral |
| `emission.py` | 19 | 0.5.13a/b/c/d Toon two-stream + spectrum-type dispatch |
| `high_res.py` | 17 | 0.5.16a audit + 0.5.16b implementation |
| `utility.py` | 35 | most setup-only / I/O — allow-listed in setup modules |
| `contributions.py` | 11 | 0.5.17b |
| `visuals.py` | 17 plotting | out of v0.5 scope |
| `supported_chemicals.py` | constants | v0 has masses; v0.5 extends to FastChem + aerosol lists |

## Iceberg disposition

POSEIDON references Iceberg via `define_model` kwargs and parameter
guards but **does not implement** the cloud model at this commit.
Per CLAUDE.md ("POSEIDON is the specification; do not re-derive
physics"), jaxPOSEIDON drops Iceberg from v0.5. The kwarg surface
accepts it; the call surfaces a descriptive `NotImplementedError`
pointing at the POSEIDON-oracle absence (NOT a jaxPOSEIDON deferral).

If a future POSEIDON commit ships an Iceberg implementation, bump
the pin and add a `0.5.NEW_ICEBERG` phase at that time.

## PyMultiNest disposition

`POSEIDON.retrieval.run_retrieval` dispatches to PyMultiNest. The
user has a separate BlackJAX NSS retrieval session; jaxPOSEIDON
v0.5 ports the **log-posterior boundary** (prior transform +
spectrum + binning + likelihood + stellar contamination
application) but does not port the sampler driver.
`jaxposeidon._retrieval.run_NSS(...)` keeps its `NotImplementedError`.
