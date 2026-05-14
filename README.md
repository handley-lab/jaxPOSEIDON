# jaxPOSEIDON

[![CI](https://github.com/handley-lab/jaxPOSEIDON/actions/workflows/ci.yml/badge.svg)](https://github.com/handley-lab/jaxPOSEIDON/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

Extending [POSEIDON](https://github.com/MartianColonist/POSEIDON) to [JAX](https://jax.readthedocs.io/). POSEIDON is the open-source heir to AURA — the canonical numerical engine for exoplanet atmospheric retrievals in the Madhusudhan-tradition pipeline. jaxPOSEIDON reuses POSEIDON's embodied numerical intelligence (opacity preprocessing, TRIDENT chord radiative transfer, MacMad17 cloud geometry, instrument convolution) and validates against POSEIDON to floating-point precision, in service of [BlackJAX nested slice sampling](https://github.com/handley-lab/blackjax) for differentiable, GPU-ready Bayesian inference.

## Usage

```python
import numpy as np
import jaxposeidon as jpo
from POSEIDON.core import (create_star, create_planet, define_model,
                            make_atmosphere, read_opacities,
                            wl_grid_constant_R)
from POSEIDON.constants import R_J, M_J, R_Sun

# Setup uses POSEIDON directly (create_star/planet/define_model/...).
star = create_star(R_Sun, 5000.0, 4.0, 0.0)
planet = create_planet("K2-18b-ish", R_J, mass=M_J, T_eq=900.0)
model = define_model("demo", ["H2", "He"], [], PT_profile="isotherm",
                     cloud_model="MacMad17", cloud_type="deck_haze")
P = np.logspace(2, -7, 100)
atm = make_atmosphere(planet, model, P, 10.0, R_J,
                       np.array([900.0]), np.array([]),
                       cloud_params=np.array([1.5, -2.0, -1.0]),
                       constant_gravity=True)
wl = wl_grid_constant_R(0.6, 5.0, 4000)
opac = read_opacities(model, wl, "opacity_sampling",
                      np.arange(700, 1110, 20), np.arange(-6, 2.2, 0.4),
                      testing=True)

# The hot path runs through jaxPOSEIDON.
spectrum = jpo.compute_spectrum(planet, star, model, atm, opac, wl)
ymodel  = jpo.bin_spectrum_to_data(spectrum, wl, data_properties)
loglike = jpo.loglikelihood(ymodel, ydata, err_data,
                             offsets_applied="single_dataset",
                             error_inflation="Line15+Piette20", ...)
```

POSEIDON setup (`create_star`, `create_planet`, `define_model`, `read_opacities`, `make_atmosphere`) is used as-is; the ported hot path is `compute_spectrum`, `bin_spectrum_to_data`, `loglikelihood`, `prior_transform`, `make_loglikelihood`, plus thin POSEIDON wrappers for `load_data` and `init_instrument`.

## Install

```bash
pip install git+https://github.com/handley-lab/jaxPOSEIDON
# POSEIDON itself is the canonical numerical oracle:
pip install git+https://github.com/MartianColonist/POSEIDON
# POSEIDON's opacity database (~70 GB; see POSEIDON docs for the download):
export POSEIDON_input_data=/path/to/POSEIDON_input_data
```

## Numerical validation

POSEIDON is the numerical oracle. 1,435 tests assert equivalence
against POSEIDON across the supported configuration surface, including
a 396-case forward-model parametric sweep over atmosphere and cloud
parameters at `atol=1e-15, rtol=1e-13` — comfortably inside the plan's
≤1 ppm target on the binned observable. The canonical Rayleigh oracle
matches POSEIDON bit-for-bit.

Six representative configurations — POSEIDON dashed, jaxPOSEIDON solid,
residual in ppm on the right:

![spectra parity](figures/parity_spectra.png)

End-to-end through the instrument model on a JWST NIRSpec PRISM-style
40-bin layout (max binned residual = 0 ppm):

![binned parity](figures/parity_binned.png)

Distribution of max-per-case |Δ(transit depth)| across the 396-case
sweep — 395 bit-exact, one case at 1.7×10⁻¹² ppm:

![sweep histogram](figures/sweep_histogram.png)

Reproduce with `python scripts/generate_{parity_figures,binning_figure,sweep_histogram}.py`.

## Scope

Supports K2-18b-style transmission retrievals: 1D background atmosphere (`PT_dim=1`, `X_dim=1`), MS09 P-T (and isotherm), isochem free chemistry over POSEIDON's supported line-list species, MacMad17 deck/haze with optional `cloud_dim=2` patchy clouds, one-offset NIRSpec-style instrument binning, Gaussian likelihood with optional offsets and Line15 / Piette20 error inflation.

The following branches are **deferred** and raise descriptive `NotImplementedError` rather than crashing with `TypeError`:

- JAX-traceable forward model and full `jax.grad` parity
- BlackJAX NSS sampler execution and the K2-18b retrieval run
- Line-by-line opacity mode
- Mie / Iceberg / eddysed cloud models
- Emission / reflection / direct / dayside / nightside spectra
- Stellar contamination
- Surfaces
- Photometric instruments (IRAC etc.)
- CLR mixing-ratio priors
- PT_penalty / Pelletier branch
- 2D / 3D atmospheres and Δ-mixing-ratio prior gating

## Running the tests

```bash
git clone https://github.com/MartianColonist/POSEIDON
PYTHONPATH=.:./POSEIDON pip install -e ".[dev]"
PYTHONPATH=.:./POSEIDON pytest tests/
```

Synthetic CIA / opacity HDF5 fixtures are built in-tempdir per-test, so the full 70 GB opacity database is **not** required for CI; only the molecular-opacity end-to-end test builds a synthetic `Opacity_database_v1.3.hdf5` with a single H2O group.

## How this library was generated

jaxPOSEIDON was produced in a single AI-assisted [Claude Code](https://www.anthropic.com/claude-code) session using a reference-guided-translation workflow: POSEIDON is the numerical oracle, every test asserts equivalence against POSEIDON, and implementation proceeded phase-by-phase with adversarial external-LLM review (`mcp__llm__review`) against the plan and the POSEIDON source after every phase. The same workflow was first demonstrated with [jaxwavelets](https://github.com/handley-lab/jaxwavelets). The full account — what worked, what required human judgment, what the agent did badly and how it was caught — is in [`CASE_STUDY.md`](CASE_STUDY.md).

## Acknowledgements

jaxPOSEIDON extends [POSEIDON](https://github.com/MartianColonist/POSEIDON) (Ryan J. MacDonald, 2022, BSD-3). POSEIDON provides the numerical reference implementation, the opacity database schema, and the instrument reference data used for validation and runtime setup.

## License

BSD-3-Clause, matching POSEIDON. Will Handley, Institute of Astronomy, Cambridge (2026).
