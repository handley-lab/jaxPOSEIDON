# External-data matrix (Phase 0.5.1 deliverable)

Every external data dependency for v0.5, with source / license /
env-var / fixture-coverage contract. CI uses synthetic fixtures
throughout; real-grid smoke tests are **env-gated**. "Required" means
v0.5 release must validate the loader against the real grid on at
least one machine, recorded in `MISMATCHES.md` (or this doc) with the
commit SHA at the time of validation. **No external grid is
redistributed** with the jaxPOSEIDON package.

| Dataset | Used by phase | Source | License / redistribution | Env var | Synthetic CI coverage | Real-grid smoke required before release | Schema-checked |
|---|---|---|---|---|---|---|---|
| POSEIDON molecular opacity DB (~70 GB; `Opacity_database_v1.3.hdf5` and friends) | 0.5.2a (`read_opacities`) + 0.5.4 (ff/bf) + 0.5.15 (LBL) | POSEIDON docs (per-species license; HITRAN/HITEMP/ExoMol) | no redistribution | `POSEIDON_input_data` | yes (synthetic H2O HDF5 with v0 schema) | **required** — minimal real-species smoke (e.g. one or two molecules + CIA pair) on at least one machine | yes |
| POSEIDON CIA HDF5 (`Opacity_database_cia.hdf5`) | already v0; extended in 0.5.4 / 0.5.15 | as above | no redistribution | `POSEIDON_input_data` | yes (synthetic H2-H2 + H2-He CIA fixture) | **required** | yes |
| POSEIDON `reference_data/` (dispatch tables: instrument sensitivity files, JWST resolution tables, surface lab albedos, FWHM dispatch tables) | 0.5.2a / 0.5.3 / 0.5.5 / 0.5.13d | POSEIDON repo (BSD-3, Ryan J. MacDonald) | redistribution allowed; jaxPOSEIDON either vendors with attribution or reads from a user-pointed copy (per-file decision at Phase 0.5.1) | `JAXPOSEIDON_REFERENCE_DATA` | yes (synthetic instrument-sensitivity + JWST-resolution + surface-lab fixtures) | required | yes |
| FastChem grids (~1 GB) | 0.5.8 | https://github.com/exoclime/FastChem (per project license; check citation requirements before use) | no redistribution | `JAXPOSEIDON_FASTCHEM_GRIDS` | yes (synthetic 4-D grid fixture covering P × T × Z × C/O × species) | required | yes |
| pysynphot ICAT grids (~100 MB) | 0.5.11 | STScI Atlas grids (Castelli–Kurucz, PHOENIX) | citation required (STScI); redistribution requires STScI check | `PYSYN_CDBS` (pysynphot's own env var) | yes (synthetic stellar grid fixture: log_g × T_eff × Z × Vmag wavelength dependence) | required | yes |
| PyMSG grids (~100 MB per grid) | 0.5.11 | https://www.astro.wisc.edu/~townsend/static.php?ref=msg | per-grid license; check redistribution case-by-case | `MSG_DIR` (PyMSG's own env var) | yes (synthetic stellar grid fixture) | required | yes |
| Zenodo aerosol DB | 0.5.12 | Zenodo DOI 10.5281/zenodo.15711943 (POSEIDON's aerosol cross-section database) | Zenodo per-record license; check redistribution before vendoring | `JAXPOSEIDON_AEROSOL_DB` | yes (synthetic aerosol HDF5 with κ_ext, κ_scat, ω₀, g for one or two species) | required | yes |
| PICASO / VIRGA eddysed output files | 0.5.14 | user-supplied (no canonical source) | n/a (user-supplied) | `JAXPOSEIDON_EDDYSED_FILES` | yes (synthetic eddysed file with κ_cloud + g + ω₀ on a representative grid) | required | yes |
| LBL HDF5 tables (~10+ GB) | 0.5.15 / 0.5.16b | POSEIDON docs + per-species line lists | no redistribution | `JAXPOSEIDON_LBL_TABLES` | yes (synthetic LBL HDF5 with one species on a small ν grid) | required | yes |

## Env-gated real-grid smoke contract

For every "required" row above:

- When the env var is **unset**, the test prints
  `SKIPPED — set ${ENV_VAR} to run` and exits 0. CI must not silently
  pass without acknowledging the skip.
- When the env var is **set**, the test loads a minimal real-grid
  payload (one species, one CIA pair, etc.) and verifies:
  - HDF5 / file schema (dataset / group / column names match POSEIDON).
  - Dispatch tables map correctly (instrument names, surface
    components, FastChem species, aerosol species).
  - One or two POSEIDON-parity smoke values agree to the
    per-component tolerance documented in `MISMATCHES.md`.
- v0.5 release gate: every "required" row has been validated on at
  least one machine. Record the validation in `MISMATCHES.md` with
  the commit SHA, the env-var path used, and the date.

## Licensing summary

- **POSEIDON `reference_data/`** (BSD-3) is the only dataset
  jaxPOSEIDON may vendor; per-file decision at Phase 0.5.1.
- **Everything else is no-redistribution**: FastChem, PyMSG,
  pysynphot Atlas, Zenodo aerosol DB, POSEIDON opacity DB, LBL
  tables. README documents the download instructions and
  citation/license expectations for each.
- **PICASO/VIRGA eddysed files are user-supplied**: no canonical
  vendor source.
