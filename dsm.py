# File: dsm.py
# Implementation of the Direct Sinusoidal Method (DSM) for crossed-cylinder superposition and sensitivity analysis.

import numpy as np

def dsm_summation(C, theta):
    """
    Compute the resultant cylinder magnitude and axis using DSM.
    
    Parameters:
    C (array): Cylinder powers (negative for convention).
    theta (array): Axes in degrees.
    
    Returns:
    C_total (float): Resultant cylinder magnitude.
    theta_total (float): Resultant axis in degrees.
    """
    theta_rad = np.deg2rad(theta)
    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))
    C_total = np.sqrt(X_net**2 + Y_net**2)
    theta_total = 0.5 * np.rad2deg(np.arctan2(Y_net, X_net))
    return C_total, theta_total

def rotational_sensitivity(C, theta, i, delta_theta_deg=1.0, C_total=None):
    """
    Compute first-order change in cylinder magnitude for perturbation in lens i (Theorem 1).
    
    Parameters:
    C (array): Cylinder powers.
    theta (array): Axes in degrees.
    i (int): Index of perturbed lens.
    delta_theta_deg (float): Perturbation in degrees.
    C_total (float): Optional precomputed C_total.
    
    Returns:
    delta_C (float): Predicted change in C_total.
    """
    theta_rad = np.deg2rad(theta)
    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))
    if C_total is None:
        C_total = np.sqrt(X_net**2 + Y_net**2)
    if C_total < 1e-10:
        raise ValueError("C_total near zero; use Corollary approximation.")
    dC_dtheta = (2 * C[i] / C_total) * (Y_net * np.cos(2 * theta_rad[i]) - X_net * np.sin(2 * theta_rad[i]))
    delta_theta_rad = np.deg2rad(delta_theta_deg)
    return dC_dtheta * delta_theta_rad

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