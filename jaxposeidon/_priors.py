"""Unit-cube prior transform.

Faithful port of POSEIDON `retrieval.py:547-1008` covering the prior
types POSEIDON itself implements:

- 'uniform'   : cube → (max-min)*cube + min
- 'gaussian'  : cube → mean + std * Φ⁻¹(cube)    [Φ⁻¹ = scipy ndtri]
- 'sine' (alpha/beta) : cube → (180/π)*2*arcsin(cube * sin(π/180 * max/2))
- 'sine' (theta_0)    : cube → (180/π)*arcsin((2*cube-1) * sin(π/180 * max/2))
- 'CLR' (mixing ratios): centred-log-ratio prior over the chemistry
  simplex (POSEIDON retrieval.py:547-594 + the dispatcher at :861-887).

Returns a NEW array; never mutates the input cube (POSEIDON's loop
mutates in place because MultiNest hands it a writable buffer — for
JAX/BlackJAX use, immutable returns are required).
"""

import numpy as np
from scipy.special import ndtri


def CLR_Prior(chem_params_drawn, limit=-12.0):
    """Centred-log-ratio prior for chemical mixing ratios.

    Bit-equivalent port of POSEIDON `retrieval.py:547-594`. Returns a
    length-(n+1) array of log10 mixing ratios on success, or
    `np.ones(n+1) * -50.0` if the draw falls outside the allowed
    simplex.
    """
    n = len(chem_params_drawn)
    prior_lower_CLR = (((n + 1) - 1.0) / (n + 1)) * (
        limit * np.log(10.0) + np.log((n + 1) - 1.0)
    )
    prior_upper_CLR = ((1.0 - (n + 1)) / (n + 1)) * (limit * np.log(10.0))

    CLR = np.zeros(shape=(n + 1))
    X = np.zeros(shape=(n + 1))

    for i in range(n):
        CLR[1 + i] = (
            chem_params_drawn[i] * (prior_upper_CLR - prior_lower_CLR) + prior_lower_CLR
        )

    if np.abs(np.sum(CLR[1 : n + 1])) <= prior_upper_CLR:
        CLR[0] = -1.0 * np.sum(CLR[1 : n + 1])
        if (np.max(CLR) - np.min(CLR)) <= (-1.0 * limit * np.log(10.0)):
            normalisation = np.sum(np.exp(CLR))
            for i in range(n + 1):
                X[i] = np.exp(CLR[i]) / normalisation
                if X[i] < 1.0e-12:
                    return np.ones(n + 1) * (-50.0)
            return np.log10(X)
        else:
            return np.ones(n + 1) * (-50.0)
    else:
        return np.ones(n + 1) * (-50.0)


def prior_transform(
    unit_cube,
    param_names,
    prior_types,
    prior_ranges,
    X_param_names=None,
    N_params_cum=None,
):
    """Map a unit hypercube vector to physical parameters.

    Mirrors POSEIDON `retrieval.py:649-887`. When any mixing-ratio
    parameter uses prior_type='CLR', `X_param_names` and `N_params_cum`
    (the cumulative-parameter boundary array from `assign_free_params`)
    must be supplied. Rejected CLR draws fill the chemistry slice with
    the sentinel value -50.0 (POSEIDON's allowed_simplex convention,
    retrieval.py:861-887); downstream likelihood checks for that.
    """
    cube = np.asarray(unit_cube, dtype=np.float64).copy()
    for i, parameter in enumerate(param_names):
        ptype = prior_types[parameter]
        prange = prior_ranges[parameter]

        if ptype == "uniform":
            min_value, max_value = prange[0], prange[1]
            cube[i] = cube[i] * (max_value - min_value) + min_value

        elif ptype == "gaussian":
            mean, std = prange[0], prange[1]
            cube[i] = mean + std * ndtri(cube[i])

        elif ptype == "sine":
            max_value = prange[1]
            if parameter in ("alpha", "beta"):
                cube[i] = (
                    (180.0 / np.pi)
                    * 2.0
                    * np.arcsin(cube[i] * np.sin((np.pi / 180.0) * (max_value / 2.0)))
                )
            elif parameter == "theta_0":
                cube[i] = (180.0 / np.pi) * np.arcsin(
                    (2.0 * cube[i] - 1.0) * np.sin((np.pi / 180.0) * (max_value / 2.0))
                )
            else:
                raise NotImplementedError(
                    f"sine prior on {parameter!r} not in v0 "
                    "(POSEIDON only uses sine for alpha/beta/theta_0)"
                )

        elif ptype == "CLR":
            pass  # Handled in a second pass below.

        else:
            raise NotImplementedError(
                f"prior_type={ptype!r} not supported (uniform/gaussian/sine/CLR)"
            )

    if "CLR" in prior_types.values():
        if X_param_names is None or N_params_cum is None:
            raise ValueError("CLR prior requires X_param_names and N_params_cum kwargs")
        chem_drawn = np.array(
            np.asarray(unit_cube, dtype=np.float64)[N_params_cum[1] : N_params_cum[2]]
        )
        limit = prior_ranges[X_param_names[0]][0]
        log_X = CLR_Prior(chem_drawn, limit)
        N_species_params = len(X_param_names)
        for i in range(N_species_params):
            cube[N_params_cum[1] + i] = log_X[1 + i]

    return cube
