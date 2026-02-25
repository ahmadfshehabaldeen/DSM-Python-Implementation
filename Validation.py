# File: validation.py
# Monte Carlo validation script for DSM.

import numpy as np
from dsm import dsm_summation, rotational_sensitivity, near_spherical_approximation

np.random.seed(42)
NUM_SYSTEMS = 10000

summation_errors = []
sensitivity_errors = []
corollary_errors = []

for _ in range(NUM_SYSTEMS):
    N = np.random.randint(2, 6)
    S = np.random.uniform(-10, 10, N)  # Not used in cylinder calc
    C = np.random.uniform(-6, 0, N)
    theta = np.random.uniform(0, 180, N)
    
    # DSM
    C_dsm, _ = dsm_summation(C, theta)
    
    # Thibos Vector (reference)
    J0 = np.sum(-C/2 * np.cos(2 * np.deg2rad(theta)))
    J45 = np.sum(-C/2 * np.sin(2 * np.deg2rad(theta)))
    C_vec = -2 * np.sqrt(J0**2 + J45**2)
    
    summation_errors.append(np.abs(np.abs(C_vec) - C_dsm))
    
    # Sensitivity (Theorem 1)
    if C_dsm > 0.1:
        i = np.random.randint(0, N)
        delta_phi_deg = 0.1
        predicted_delta_C = rotational_sensitivity(C, theta, i, delta_phi_deg, C_dsm)
        
        theta_new = theta.copy()
        theta_new[i] += delta_phi_deg
        C_new, _ = dsm_summation(C, theta_new)
        actual_delta_C = C_new - C_dsm
        sensitivity_errors.append(np.abs(predicted_delta_C - actual_delta_C))

# Corollary validation
for _ in range(1000):
    C1 = np.random.uniform(-4, -1)
    phi_deg = np.random.uniform(0.1, 3.0)
    # Simulate perfect correction: C2 = -C1, theta2 = phi
    C = np.array([C1, -C1])
    theta = np.array([0, phi_deg])
    C_actual, _ = dsm_summation(C, theta)
    C_pred = near_spherical_approximation(C1, phi_deg)
    corollary_errors.append(np.abs(C_actual - C_pred))

print(f"Max Summation Error: {np.max(summation_errors):.2e}")
print(f"Max Sensitivity Error: {np.max(sensitivity_errors):.2e}")
print(f"Max Corollary Error: {np.max(corollary_errors):.2e}")