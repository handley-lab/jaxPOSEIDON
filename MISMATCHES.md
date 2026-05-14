# jaxposeidon ↔ POSEIDON mismatches

Documented numerical differences between jaxposeidon's v0 port and
POSEIDON reference outputs. Each entry should include: POSEIDON
file:line, component, observed delta, root cause, and whether it's
acceptable (documented mismatch) or a defect to fix.

## Open numerical mismatches

*(none — v0 forward model matches POSEIDON bit-exactly on the canonical
Rayleigh oracle and to `atol=1e-15, rtol=1e-13` on the 396-case
parametric sweep.)*

## Known non-numerical issues

- **Test-suite order pollution**: 4 tests pass in isolation but fail
  when the full suite runs in pytest's default collection order
  (Madhu pressure-ordering rejection, two extinction parametric cases,
  one combinatorial likelihood case). Likely cause: shared mutable
  state in POSEIDON globals after early `read_opacities(...)` calls,
  or pytest's seed reuse across hash-seeded RNGs. Workaround: run
  failing tests in isolation. Tracked as Phase 11 / v1 work.

## Resolved mismatches

*(none.)*
