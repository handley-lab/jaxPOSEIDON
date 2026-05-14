# jaxposeidon ↔ POSEIDON mismatches

Documented numerical differences between jaxposeidon's v0 port and
POSEIDON reference outputs.

## Open numerical mismatches

*(none — v0 forward model matches POSEIDON bit-exactly on the canonical
Rayleigh oracle and to `atol=1e-15, rtol=1e-13` on the 396-case
parametric sweep; full test suite is 1435 passed, 1 skipped (the
real-DB conditional).)*

## Resolved

- **Phase 4 extinction parametric tolerance**: POSEIDON's numba reduction
  order produces ULP-scale residuals (~4e-25 absolute) on the
  `(haze=1, deck=1)` configuration. The `atol=0, rtol=0` assertion was
  loosened to `atol=1e-22, rtol=1e-13`, well inside FP precision.
- **Phase 2 Madhu signature**: `POSEIDON.atmosphere.profiles` added
  positional kwargs (`T_input`, `X_input`, `P_param_set`,
  `log_P_slope_phot`, `log_P_slope_arr`, `Na_K_fixed_ratio`); the
  rejection test now passes the additional arguments.
