"""Angular grids for 1D background atmospheres.

Faithful port of POSEIDON's `geometry.py:12-253` for the v0 envelope:
`Atmosphere_dimension == 1`. Higher-dimensionality atmospheres (2D D-N,
2D E-M, 3D) raise NotImplementedError.
"""

import numpy as np


def atmosphere_regions(Atmosphere_dimension, TwoD_type=None,
                       N_slice_EM=2, N_slice_DN=2):
    """Number of azimuthal sectors / zenith zones for the background atmosphere.

    Mirrors POSEIDON `geometry.py:12-87` for the 1D case.
    """
    if Atmosphere_dimension != 1:
        raise NotImplementedError(
            f"v0 only supports Atmosphere_dimension=1 (got {Atmosphere_dimension})"
        )
    return 1, 1


def angular_grids(Atmosphere_dimension, TwoD_type=None,
                  N_slice_EM=2, N_slice_DN=2,
                  alpha=0.0, beta=0.0,
                  sharp_DN_transition=False, sharp_EM_transition=False):
    """phi, theta, phi_edge, theta_edge, dphi, dtheta for 1D atmospheres.

    Mirrors POSEIDON `geometry.py:90-253`. For 1D, edges are placed in the
    Evening-Morning / Day-Night equatorial planes at +/- π/2.
    """
    if Atmosphere_dimension != 1:
        raise NotImplementedError(
            f"v0 only supports Atmosphere_dimension=1 (got {Atmosphere_dimension})"
        )
    phi_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
    theta_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
    dphi = np.diff(phi_edge)
    dtheta = np.diff(theta_edge)
    phi = -np.pi / 2.0 + np.cumsum(dphi) - (dphi / 2.0)
    theta = -np.pi / 2.0 + np.cumsum(dtheta) - (dtheta / 2.0)
    return phi, theta, phi_edge, theta_edge, dphi, dtheta
