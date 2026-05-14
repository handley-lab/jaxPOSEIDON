"""Phase 10a: unit-cube prior transform parity with POSEIDON.

v0 prior types: uniform, gaussian, sine (for alpha/beta/theta_0).
CLR is v1 (and raises NotImplementedError).
"""

import numpy as np
import pytest

from jaxposeidon._priors import prior_transform


def _poseidon_prior_inplace(cube, param_names, prior_types, prior_ranges):
    """Direct line-for-line replication of POSEIDON retrieval.py:649-708
    for Atmosphere_dimension=1, non-CLR priors."""
    from scipy.special import ndtri
    cube = np.array(cube, dtype=np.float64).copy()
    for i, parameter in enumerate(param_names):
        ptype = prior_types[parameter]
        prange = prior_ranges[parameter]
        if ptype == "uniform":
            cube[i] = cube[i] * (prange[1] - prange[0]) + prange[0]
        elif ptype == "gaussian":
            cube[i] = prange[0] + prange[1] * ndtri(cube[i])
        elif ptype == "sine":
            max_value = prange[1]
            if parameter in ("alpha", "beta"):
                cube[i] = (180.0 / np.pi) * 2.0 * np.arcsin(
                    cube[i] * np.sin((np.pi / 180.0) * (max_value / 2.0))
                )
            elif parameter == "theta_0":
                cube[i] = (180.0 / np.pi) * np.arcsin(
                    (2.0 * cube[i] - 1.0)
                    * np.sin((np.pi / 180.0) * (max_value / 2.0))
                )
    return cube


def test_prior_transform_uniform():
    param_names = ["T", "R_p", "log_X_H2O"]
    pt = {p: "uniform" for p in param_names}
    pr = {"T": [400.0, 2500.0], "R_p": [0.8, 1.2],
          "log_X_H2O": [-12.0, -1.0]}
    cube = np.array([0.0, 0.5, 1.0])
    out = prior_transform(cube, param_names, pt, pr)
    np.testing.assert_array_equal(
        out, _poseidon_prior_inplace(cube, param_names, pt, pr),
    )


def test_prior_transform_gaussian():
    param_names = ["log_g"]
    pt = {"log_g": "gaussian"}
    pr = {"log_g": [3.5, 0.2]}  # mean=3.5, std=0.2
    for u in [0.05, 0.25, 0.5, 0.75, 0.95]:
        cube = np.array([u])
        out = prior_transform(cube, param_names, pt, pr)
        np.testing.assert_allclose(
            out, _poseidon_prior_inplace(cube, param_names, pt, pr),
            atol=0, rtol=0,
        )


def test_prior_transform_sine_alpha_beta_theta0():
    param_names = ["alpha", "beta", "theta_0"]
    pt = {p: "sine" for p in param_names}
    pr = {"alpha": [0.0, 60.0], "beta": [0.0, 30.0],
          "theta_0": [-90.0, 90.0]}
    cube = np.array([0.1, 0.4, 0.7])
    out = prior_transform(cube, param_names, pt, pr)
    np.testing.assert_array_equal(
        out, _poseidon_prior_inplace(cube, param_names, pt, pr),
    )


def test_prior_transform_does_not_mutate_input():
    cube = np.array([0.3, 0.7])
    pt = {"a": "uniform", "b": "uniform"}
    pr = {"a": [0.0, 10.0], "b": [-1.0, 1.0]}
    _ = prior_transform(cube, ["a", "b"], pt, pr)
    np.testing.assert_array_equal(cube, np.array([0.3, 0.7]))


def test_prior_transform_rejects_CLR():
    with pytest.raises(NotImplementedError, match="CLR"):
        prior_transform(np.array([0.5]), ["log_X"],
                         {"log_X": "CLR"}, {"log_X": [-12.0, -1.0]})


@pytest.mark.parametrize("ptype", ["log_uniform", "Jeffreys", "delta"])
def test_prior_transform_rejects_unknown_prior_types(ptype):
    with pytest.raises(NotImplementedError, match="prior_type"):
        prior_transform(np.array([0.5]), ["x"],
                         {"x": ptype}, {"x": [0.0, 1.0]})
