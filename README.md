# jaxposeidon

JAX-native port of [POSEIDON](https://github.com/MartianColonist/POSEIDON)'s
transmission-spectroscopy forward model.

**Status: Phase 0 (scaffolding).** No functional code yet — see
`~/.claude/plans/let-s-get-going-shimmering-parnas.md` for the
12-phase plan and per-phase tolerance targets.

## Why

POSEIDON is the open-source heir to AURA (Ryan MacDonald, formerly
Madhusudhan group, github.com/MartianColonist/POSEIDON). It is the
canonical numerical engine for exoplanet atmospheric retrievals in the
Madhusudhan-tradition pipeline. It uses pre-computed HDF5 opacity grids,
TRIDENT chord radiative transfer, and PyMultiNest for inference.

POSEIDON itself is numpy + scipy + numba. This port preserves the
forward-model numerics (validated against POSEIDON to component-specific
tolerances) while adding:

- **JAX differentiability** — `jax.grad` of `log_likelihood`, enabling
  HMC, variational inference, and normalising-flow posteriors.
- **JAX composability** — `vmap` over parameter ensembles, `jit`
  compilation, optional GPU execution.
- **BlackJAX nested slice sampling** as the default sampler
  (handley-lab fork v0.1.0-beta), replacing PyMultiNest.

## v0 scope

See `CLAUDE.md` and the plan file. v0 supports K2-18 b-style retrievals
(MS09 P-T, isochem free chemistry, MacMad17 deck/haze with optional
`cloud_dim=2` patchy clouds, spectroscopic instrument binning, one-offset
NIRSpec). Stellar contamination, Mie clouds, equilibrium chemistry,
emission/reflection, LBL mode, and 2D/3D atmospheres are deferred.

## Install

```bash
pip install jaxposeidon
# plus POSEIDON itself (as numerical oracle for tests):
pip install git+https://github.com/MartianColonist/POSEIDON
# plus POSEIDON's opacity database (70+ GB; see POSEIDON docs).
export POSEIDON_input_data=/path/to/POSEIDON_input_data
```

## Faithfulness

This port is validated against POSEIDON at per-component tolerances
(see plan). It is **not** bit-identical to POSEIDON — JAX/XLA reorder
some reductions and POSEIDON uses scipy PCHIP / numba paths that we
mirror in pure JAX with sub-percent residuals on cross-sections and
≤1 ppm on binned transmission spectra.

## Attribution

POSEIDON: Ryan J. MacDonald (2022, BSD-3).
jaxposeidon: Will Handley, Institute of Astronomy, Cambridge (2026,
BSD-3).
