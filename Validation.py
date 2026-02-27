"""
Validation script for DSM
"""

import numpy as np
from dsm import dsm_summation


def classical_method(C, theta_deg):

    theta_rad = np.deg2rad(theta_deg)

    X = np.sum(C * np.cos(theta_rad))
    Y = np.sum(C * np.sin(theta_rad))

    magnitude = np.sqrt(X**2 + Y**2)
    angle = np.rad2deg(np.arctan2(Y, X))

    return magnitude, angle


def run_validation():

    np.random.seed(42)

    max_error = 0

    for _ in range(1000):

        C = np.random.uniform(1, 100, 10)
        theta = np.random.uniform(0, 180, 10)

        C_dsm, _ = dsm_summation(C, theta)
        C_classical, _ = classical_method(C, theta)

        error = abs(C_dsm - C_classical)

        max_error = max(max_error, error)

    print("Max error:", max_error)


if __name__ == "__main__":
    run_validation()print(f"Max Corollary Error: {np.max(corollary_errors):.2e}")
