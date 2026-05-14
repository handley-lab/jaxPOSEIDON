"""Angular grids for 1D / 2D / 3D atmospheres.

Port of POSEIDON's `geometry.py:12-253` covering the full
`Atmosphere_dimension in {1, 2, 3}` surface.
"""

import numpy as np


def atmosphere_regions(
    Atmosphere_dimension, TwoD_type=None, N_slice_EM=2, N_slice_DN=2
):
    """Number of azimuthal sectors / zenith zones for the background atmosphere.

    Bit-equivalent to POSEIDON `geometry.py:12-87`.
    """
    if Atmosphere_dimension == 1:
        return 1, 1
    elif Atmosphere_dimension == 2:
        if TwoD_type == "E-M":
            if N_slice_EM < 0 or N_slice_EM % 2 != 0:
                raise Exception("Error: N_slice_EW must be an even integer.")
            return 2 + N_slice_EM, 1
        elif TwoD_type == "D-N":
            if N_slice_DN < 0 or N_slice_DN % 2 != 0:
                raise Exception("Error: N_slice_DN must be an even integer.")
            return 1, 2 + N_slice_DN
        else:
            raise Exception(f"Error: '{TwoD_type}' is not a valid 2D model type.")
    elif Atmosphere_dimension == 3:
        if (
            N_slice_EM < 0
            or N_slice_EM % 2 != 0
            or N_slice_DN < 0
            or N_slice_DN % 2 != 0
        ):
            raise Exception("Error: N_slice_EW and N_slice_DN must be even integers.")
        return 2 + N_slice_EM, 2 + N_slice_DN
    elif Atmosphere_dimension == 4:
        raise Exception("Error: Planets can't be tesseracts!")
    else:
        raise Exception("Error: Invalid dimensionality for model atmosphere.")


def angular_grids(
    Atmosphere_dimension,
    TwoD_type=None,
    N_slice_EM=2,
    N_slice_DN=2,
    alpha=0.0,
    beta=0.0,
    sharp_DN_transition=False,
    sharp_EM_transition=False,
):
    """Sector / zone angular grids for 1D / 2D / 3D atmospheres.

    Bit-equivalent to POSEIDON `geometry.py:90-253`.
    """
    alpha_rad = alpha * (np.pi / 180.0)
    beta_rad = beta * (np.pi / 180.0)

    if Atmosphere_dimension == 1:
        phi_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
        theta_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
    elif Atmosphere_dimension == 2:
        if TwoD_type == "E-M":
            theta_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
            if sharp_EM_transition:
                phi_edge = np.array([-np.pi / 2.0, 0.0, np.pi / 2.0])
            else:
                phi_edge = np.array([-np.pi / 2.0])
                dphi_term = alpha_rad / N_slice_EM
                phi_edge = np.append(
                    phi_edge,
                    -0.5 * alpha_rad + np.arange(N_slice_EM + 1) * dphi_term,
                )
                phi_edge = np.append(phi_edge, np.array([np.pi / 2.0]))
        elif TwoD_type == "D-N":
            phi_edge = np.array([-np.pi / 2.0, np.pi / 2.0])
            if sharp_DN_transition:
                theta_edge = np.array([-np.pi / 2.0, 0.0, np.pi / 2.0])
            else:
                theta_edge = np.array([-np.pi / 2.0])
                dtheta_term = beta_rad / N_slice_DN
                theta_edge = np.append(
                    theta_edge,
                    -0.5 * beta_rad + np.arange(N_slice_DN + 1) * dtheta_term,
                )
                theta_edge = np.append(theta_edge, np.array([np.pi / 2.0]))
    elif Atmosphere_dimension == 3:
        if sharp_DN_transition:
            theta_edge = np.array([-np.pi / 2.0, 0.0, np.pi / 2.0])
        else:
            theta_edge = np.array([-np.pi / 2.0])
            dtheta_term = beta_rad / N_slice_DN
            theta_edge = np.append(
                theta_edge,
                -0.5 * beta_rad + np.arange(N_slice_DN + 1) * dtheta_term,
            )
            theta_edge = np.append(theta_edge, np.array([np.pi / 2.0]))

        if sharp_EM_transition:
            phi_edge = np.array([-np.pi / 2.0, 0.0, np.pi / 2.0])
        else:
            phi_edge = np.array([-np.pi / 2.0])
            dphi_term = alpha_rad / N_slice_EM
            phi_edge = np.append(
                phi_edge,
                -0.5 * alpha_rad + np.arange(N_slice_EM + 1) * dphi_term,
            )
            phi_edge = np.append(phi_edge, np.array([np.pi / 2.0]))
    else:
        raise Exception("Error: Invalid dimensionality for model atmosphere.")

    dphi = np.diff(phi_edge)
    dtheta = np.diff(theta_edge)
    phi = -np.pi / 2.0 + np.cumsum(dphi) - (dphi / 2.0)
    theta = -np.pi / 2.0 + np.cumsum(dtheta) - (dtheta / 2.0)
    return phi, theta, phi_edge, theta_edge, dphi, dtheta
