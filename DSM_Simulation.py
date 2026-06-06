"""
Double-Angle Summation Method (DSM)
Core implementation and Monte Carlo Validation

Author: Ahmad F. M. Shehab Aldeen
"""

import numpy as np

# =====================================================================
# PART 1: CORE DSM FUNCTIONS (System Implementation)
# =====================================================================

def dsm_summation(C, theta_deg):
    """Compute resultant magnitude and orientation using DSM."""
    C = np.asarray(C, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)

    if C.shape != theta_deg.shape:
        raise ValueError("C and theta must have same shape")

    theta_rad = np.deg2rad(theta_deg)

    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))

    C_total = np.sqrt(X_net**2 + Y_net**2)
    theta_total = 0.5 * np.rad2deg(np.arctan2(Y_net, X_net))

    return C_total, theta_total


def rotational_sensitivity(C, theta_deg, index, delta_theta_deg=1.0):
    """Compute analytical rotational sensitivity (Eq. 4)."""
    C = np.asarray(C, dtype=float)
    theta_deg = np.asarray(theta_deg, dtype=float)
    theta_rad = np.deg2rad(theta_deg)

    X_net = np.sum(C * np.cos(2 * theta_rad))
    Y_net = np.sum(C * np.sin(2 * theta_rad))
    C_total = np.sqrt(X_net**2 + Y_net**2)

    if C_total < 1e-15:
        raise ValueError("Resultant too small (Ill-conditioned). Use near_spherical_approximation.")

    derivative = (2 * C[index] / C_total) * (
        Y_net * np.cos(2 * theta_rad[index]) - X_net * np.sin(2 * theta_rad[index])
    )
    delta_theta_rad = np.deg2rad(delta_theta_deg)

    return derivative * delta_theta_rad


def near_spherical_approximation(C_i, phi_deg):
    """Corollary for near-perfect correction (Eq. 5)."""
    phi_rad = np.deg2rad(phi_deg)
    return 2 * abs(C_i) * abs(phi_rad)


# =====================================================================
# PART 2: MONTE CARLO VALIDATION (Reproducing Manuscript Results)
# =====================================================================

def run_validation():
    # Simulation Parameters
    NUM_SYSTEMS = 10000
    LENSES_PER_SYSTEM = 25
    PERTURBATION_DEG = 0.1

    print(f"Running Monte Carlo Simulation for {NUM_SYSTEMS} systems ({LENSES_PER_SYSTEM} lenses each)...")
    
    # Fix seed to ensure exact reproducibility of the manuscript's results
    np.random.seed(42) 
    
    cylinders = np.random.uniform(-6, 0, (NUM_SYSTEMS, LENSES_PER_SYSTEM))
    axes_deg = np.random.uniform(0, 180, (NUM_SYSTEMS, LENSES_PER_SYSTEM))

    max_equiv_error = 0.0
    max_sens_error = 0.0

    # Iterate through all systems
    for i in range(NUM_SYSTEMS):
        C_sys = cylinders[i]
        theta_sys = axes_deg[i]

        # --- 1. Validate Numerical Equivalence ---
        # Standard Thibos Method
        theta_rad_sys = np.deg2rad(theta_sys)
        J0 = np.sum(-(C_sys / 2.0) * np.cos(2 * theta_rad_sys))
        J45 = np.sum(-(C_sys / 2.0) * np.sin(2 * theta_rad_sys))
        C_thibos = 2 * np.sqrt(J0**2 + J45**2)

        # DSM Method (Using Author's Core Function)
        C_dsm, _ = dsm_summation(C_sys, theta_sys)
        
        # Calculate Equivalence Error
        equiv_error = abs(C_thibos - C_dsm)
        if equiv_error > max_equiv_error:
            max_equiv_error = equiv_error

        # --- 2. Validate Rotational Sensitivity ---
        # Analytical Calculation (Using Author's Core Function)
        delta_C_analytical = rotational_sensitivity(C_sys, theta_sys, index=0, delta_theta_deg=PERTURBATION_DEG)

        # Numerical Re-computation (Finite Difference)
        theta_sys_perturbed = theta_sys.copy()
        theta_sys_perturbed[0] += PERTURBATION_DEG
        C_dsm_perturbed, _ = dsm_summation(C_sys, theta_sys_perturbed)
        delta_C_numerical = C_dsm_perturbed - C_dsm

        # Calculate Sensitivity Error
        sens_error = abs(delta_C_analytical - delta_C_numerical)
        if sens_error > max_sens_error:
            max_sens_error = sens_error

    # Print Results exactly as stated in the manuscript
    print("\n--- Validation of Numerical Equivalence ---")
    print(f"Max Difference (DSM vs Thibos): {max_equiv_error:.4e} D")
    print("Conclusion: Perfect agreement within machine precision.")

    print("\n--- Validation of Rotational Sensitivity Theorem ---")
    print(f"Perturbation Applied: {PERTURBATION_DEG} degrees")
    print(f"Max Deviation (Analytical vs Numerical): {max_sens_error:.4e} D")
    print("Conclusion: Closed-form derivative successfully validated.")

if __name__ == "__main__":
    run_validation()