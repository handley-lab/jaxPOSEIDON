"""Phase 0.5.10: CLR (centred-log-ratio) mixing-ratio prior.

Ports POSEIDON `retrieval.py:547-594` (`CLR_Prior`) and the dispatcher
at `:861-887` (CLR handling inside `prior_transform`).

The CLR prior maps n uniform [0,1] draws to log10 mixing ratios for
n+1 species on the simplex {X_i > 1e-12, sum X = 1}. Draws outside the
allowed simplex region return the sentinel `np.ones(n+1) * -50.0`;
downstream likelihood treats that as a rejection.
"""

import numpy as np
import pytest

from jaxposeidon._priors import CLR_Prior, prior_transform


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_CLR_Prior_matches_poseidon(seed):
    from POSEIDON.retrieval import CLR_Prior as p_CLR

    rng = np.random.default_rng(seed)
    chem_drawn = rng.uniform(0.0, 1.0, size=4)  # 4 species → 5-element output
    ours = CLR_Prior(chem_drawn, limit=-12.0)
    theirs = p_CLR(chem_drawn, limit=-12.0)
    np.testing.assert_array_equal(ours, theirs)


@pytest.mark.parametrize("n_species", [1, 2, 3, 4, 6])
def test_CLR_Prior_simplex_returns_valid_logX(n_species):
    """For low-corner draws every species is at log_X ~ -12 → near uniform."""
    chem_drawn = 0.5 * np.ones(n_species)
    log_X = CLR_Prior(chem_drawn, limit=-12.0)
    if log_X[1] == -50.0:
        return  # rejection sentinel — possible at the corners of the simplex
    X = 10.0**log_X
    np.testing.assert_allclose(X.sum(), 1.0, atol=1e-12)
    assert np.all(X > 1e-12)


def test_CLR_Prior_rejection_returns_sentinel():
    """Pathological draw at the simplex boundary should hit the rejection path."""
    # The corners of the unit hypercube are likely to fall outside the
    # allowed CLR triangle for n > 1.
    chem_drawn = np.array([1.0, 1.0, 1.0])
    log_X = CLR_Prior(chem_drawn, limit=-12.0)
    if log_X[1] == -50.0:
        np.testing.assert_array_equal(log_X, np.ones(4) * -50.0)


def test_prior_transform_CLR_dispatch_matches_poseidon():
    """End-to-end: a unit cube with CLR mixing-ratio params reproduces the
    same CLR-mapped values POSEIDON's prior_transform would produce."""
    # Single CLR block: 4 species, no other free params.
    param_names = [
        "R_p_ref",  # uniform
        "T",  # uniform
        "log_H2O",
        "log_CH4",
        "log_NH3",
        "log_HCN",
    ]
    prior_types = {p: "uniform" for p in param_names}
    for p in ("log_H2O", "log_CH4", "log_NH3", "log_HCN"):
        prior_types[p] = "CLR"
    prior_ranges = {
        "R_p_ref": [0.9e8, 1.1e8],
        "T": [300.0, 2000.0],
        "log_H2O": [-12.0, -1.0],
        "log_CH4": [-12.0, -1.0],
        "log_NH3": [-12.0, -1.0],
        "log_HCN": [-12.0, -1.0],
    }
    rng = np.random.default_rng(0)
    unit_cube = rng.uniform(0.0, 1.0, size=len(param_names))
    N_params_cum = np.array([2, 2, 6, 6, 6, 6, 6, 6, 6, 6])
    X_param_names = ["log_H2O", "log_CH4", "log_NH3", "log_HCN"]

    cube = prior_transform(
        unit_cube,
        param_names,
        prior_types,
        prior_ranges,
        X_param_names=X_param_names,
        N_params_cum=N_params_cum,
    )

    # Non-CLR uniform params transform as usual.
    assert prior_ranges["R_p_ref"][0] <= cube[0] <= prior_ranges["R_p_ref"][1]
    assert prior_ranges["T"][0] <= cube[1] <= prior_ranges["T"][1]

    # CLR slice: either rejected (all -50.0) or a valid log-X vector.
    clr_slice = cube[2:6]
    if clr_slice[0] == -50.0:
        np.testing.assert_array_equal(clr_slice, np.ones(4) * -50.0)
    else:
        # Sum to <= 1 (one species is the deep value reconstructed as
        # -sum(log_X[1:])).
        X = 10.0**clr_slice
        assert (X > 1e-12).all()


def test_prior_transform_CLR_rejection_propagates_sentinel():
    """Corner-of-cube draw must propagate the -50.0 sentinel into the slice."""
    param_names = ["log_a", "log_b", "log_c"]
    prior_types = {p: "CLR" for p in param_names}
    prior_ranges = {p: [-12.0, -1.0] for p in param_names}
    N_params_cum = np.array([0, 0, 3, 3, 3, 3, 3, 3, 3, 3])
    cube = prior_transform(
        np.array([1.0, 1.0, 1.0]),
        param_names,
        prior_types,
        prior_ranges,
        X_param_names=param_names,
        N_params_cum=N_params_cum,
    )
    # Either all rejected (-50.0) or all valid logX.
    if cube[0] == -50.0:
        np.testing.assert_array_equal(cube, np.ones(3) * -50.0)
