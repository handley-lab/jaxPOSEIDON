"""Unit-cube prior transform — v0 port.

Faithful port of POSEIDON `retrieval.py:649-1008` filtered to the v0
envelope (Atmosphere_dimension=1, no CLR, no surface_percentage CLR,
no 2D/3D delta priors).

v0 prior types per `retrieval.py:663-688`:
  - 'uniform'   : cube → (max-min)*cube + min
  - 'gaussian'  : cube → mean + std * Φ⁻¹(cube)    [Φ⁻¹ = scipy ndtri]
  - 'sine' (alpha/beta) : cube → (180/π)*2*arcsin(cube * sin(π/180 * max/2))
  - 'sine' (theta_0)    : cube → (180/π)*arcsin((2*cube-1) * sin(π/180 * max/2))

CLR (`retrieval.py:861-887`), 2D/3D mixing-ratio gradient priors
(`:707-859`), and surface_percentage CLR (`:889-1008`) are v1.

Returns a NEW array; never mutates the input cube (POSEIDON's loop
mutates in place because MultiNest hands it a writable buffer — for
JAX/BlackJAX use, immutable returns are required).
"""

import numpy as np
from scipy.special import ndtri


def prior_transform(unit_cube, param_names, prior_types, prior_ranges):
    """Map a unit hypercube vector to physical parameters.

    Mirrors POSEIDON `retrieval.py:649-708` for Atmosphere_dimension=1
    and non-CLR priors.
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
            raise NotImplementedError(
                "CLR mixing-ratio prior (retrieval.py:861-887) is v1"
            )

        else:
            raise NotImplementedError(
                f"prior_type={ptype!r} not in v0 (uniform/gaussian/sine only)"
            )
    return cube
