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

## Resolved

- **Phase 4 extinction parametric tolerance**: POSEIDON's numba reduction
  order produces ULP-scale residuals (~4e-25 absolute) on the
  `(haze=1, deck=1)` configuration. The `atol=0, rtol=0` assertion was
  loosened to `atol=1e-22, rtol=1e-13`, well inside FP precision.
- **Phase 2 Madhu signature**: `POSEIDON.atmosphere.profiles` added
  positional kwargs (`T_input`, `X_input`, `P_param_set`,
  `log_P_slope_phot`, `log_P_slope_arr`, `Na_K_fixed_ratio`); the
  rejection test now passes the additional arguments.
