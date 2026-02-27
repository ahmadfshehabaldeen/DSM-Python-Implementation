"""
DSM Implementation
Double-Angle Summation Method (DSM)

Author: Ahmad Shehab
"""

import numpy as np


def dsm_summation(C, theta_deg):
    """
    Compute resultant magnitude and orientation using DSM.

    Parameters
    ----------
    C : array-like
        Magnitudes
    theta_deg : array-like
        Angles in degrees

    Returns
    -------
    C_total : float
        Resultant magnitude
    theta_total : float
        Resultant angle in degrees
    """

    C = np.asarray(C, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)

    theta_rad = np.deg2rad(theta_deg)

    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))

    C_total = np.sqrt(X_net**2 + Y_net**2)

    theta_total = 0.5 * np.rad2deg(np.arctan2(Y_net, X_net))

    return C_total, theta_total


def rotational_sensitivity(C, theta_deg, i, delta_theta_deg=1.0):
    """
    Compute sensitivity to angular perturbation.

    Parameters
    ----------
    C : array-like
    theta_deg : array-like
    i : index of perturbed element
    delta_theta_deg : float

    Returns
    -------
    delta_C : float
    """

    C = np.asarray(C, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)

    theta_rad = np.deg2rad(theta_deg)

    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))

    C_total = np.sqrt(X_net**2 + Y_net**2)

    if C_total < 1e-15:
        raise ValueError("Resultant magnitude too small")

    derivative = (
        2 * C[i] / C_total
    ) * (
        Y_net * np.cos(2 * theta_rad[i])
        - X_net * np.sin(2 * theta_rad[i])
    )

    delta_theta_rad = np.deg2rad(delta_theta_deg)

    delta_C = derivative * delta_theta_rad

    return delta_C


def near_spherical_approximation(C_i, phi_deg):
    """
    Near-spherical approximation corollary.
    """

    phi_rad = np.deg2rad(phi_deg)

    return 2 * abs(C_i) * abs(phi_rad)


if __name__ == "__main__":
    # Example usage

    C = [10, 5, 7]
    theta = [30, 60, 120]

    C_total, theta_total = dsm_summation(C, theta)

    print("Resultant magnitude:", C_total)
    print("Resultant angle:", theta_total)
def near_spherical_approximation(C_i, phi_deg):
    """
    Corollary 1: Induced cylinder for small misalignment near perfect correction.
    
    Parameters:
    C_i (float): Magnitude of cylinder (absolute).
    phi_deg (float): Misalignment in degrees.
    
    Returns:
    delta_C (float): Approximate induced cylinder.
    """
    phi_rad = np.deg2rad(phi_deg)

    return 2 * np.abs(C_i) * np.abs(phi_rad)
