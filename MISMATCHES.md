# jaxposeidon ↔ POSEIDON mismatches

Documented numerical differences between jaxposeidon's v0 port and
POSEIDON reference outputs.

## Open numerical mismatches

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
