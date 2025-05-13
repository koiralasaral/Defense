import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Define Material attenuation coefficients (1/cm)
# For each radiation type, we choose approximate values. 
# (These are exemplary numbers.)
# ============================================================
materials = {
    "Lead": {
        "alpha": 460.0,    # very high absorption: a few 0.01 cm is enough
        "beta": 0.8,       # moderately absorbing for beta electrons
        "gamma": 1.2,      # gamma
        "high_gamma": 0.5  # experimental high–energy gamma that is more penetrating
    },
    "Cement": {
        "alpha": 46.0,     # one order of magnitude lower than lead
        "beta": 0.4,
        "gamma": 0.2,
        "high_gamma": 0.1
    },
    "Reflective": {       # Assume a polished aluminum–like material
        "alpha": 600.0,
        "beta": 0.5,
        "gamma": 0.2,
        "high_gamma": 0.15
    }
}

radiation_types = ['alpha', 'beta', 'gamma', 'high_gamma']

# ============================================================
# Calculate and print the required thickness for 99% absorption
# (i.e. transmission I/I0 = 0.01).
# ============================================================
print("=== Required Thickness (cm) for 99% Attenuation (I/I0=0.01) ===")
for mat in materials:
    print(f"Material: {mat}")
    for rad in radiation_types:
        mu = materials[mat][rad]
        t_req = 4.60517 / mu  # thickness required in cm
        print(f"  {rad:10s}: t_req = {t_req:7.4f} cm (mu={mu})")
    print()

# ============================================================
# Plot the decay (attenuation) curves for each material.
# For each radiation type, we plot I(x) = exp(-mu*x) 
# over a thickness range from 0 to 1.2*t_req.
# Also, print intermediate values.
# ============================================================
for mat in materials:
    plt.figure(figsize=(8, 6))
    print(f"--- {mat}: Intermediate Intensity Values ---")
    for rad in radiation_types:
        mu = materials[mat][rad]
        t_req = 4.60517 / mu  # thickness for 1% transmission
        # Generate thickness values from 0 to 1.2*t_req
        x = np.linspace(0, 1.2*t_req, 100)
        intensity = np.exp(-mu * x)
        plt.plot(x, intensity, label=f"{rad} (t_req={t_req:.3f} cm)")
        
        # Print intermediate intensity values at selected fractions of t_req
        print(f"{mat} - {rad}:")
        fractions = [0, 0.25, 0.5, 0.75, 1.0]
        for frac in fractions:
            thickness_val = frac * t_req
            I_val = np.exp(-mu * thickness_val)
            print(f"   Thickness = {thickness_val:7.4f} cm -> I/I0 = {I_val:7.4f}")
        print()
    
    plt.title(f"Radiation Attenuation in {mat}")
    plt.xlabel("Thickness (cm)")
    plt.ylabel("Relative Intensity (I/I0)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# Define a function to simulate deflection for a given radiation type.
# We assume that the deflection angle (in radians) has a Gaussian
# distribution with a standard deviation that increases as:
#     sigma = k * sqrt(t)
# where t is the shield thickness (in cm) and k is an ad-hoc constant.
# We then propagate these rays for a fixed distance to compute their 
# lateral displacement.
# ============================================================
def simulate_deflection(t_sim, num_rays, k):
    # t_sim: selected thickness in cm (a representative value)
    # k: proportionality constant for √(thickness) dependence (in rad/sqrt(cm))
    sigma_deflection = k * np.sqrt(t_sim)
    angles = np.random.normal(0, sigma_deflection, num_rays)  # in radians
    # Assume the rays travel a distance L after emerging from the shield:
    L = 20  # cm
    lateral_offsets = L * np.tan(angles)
    return angles, lateral_offsets, sigma_deflection

# ============================================================
# Simulate and plot deflection (rays) for beta and gamma radiation.
# (For alpha, the range is too short and deflection is negligible.)
# We simulate for each material using half the required thickness for that radiation.
# ============================================================
for mat in materials:
    # --- For Beta Radiation ---
    mu_beta = materials[mat]['beta']
    t_req_beta = 4.60517 / mu_beta
    t_sim_beta = t_req_beta / 2  # choose a representative thickness (cm)
    k_beta = 0.2  # chosen constant (beta particles are deflected more)
    angles_beta, offsets_beta, sigma_beta = simulate_deflection(t_sim_beta, num_rays=100, k=k_beta)
    
    plt.figure(figsize=(6, 4))
    for off in offsets_beta:
        plt.arrow(0, 0, 20, off, head_width=0.5, head_length=0.5, fc='blue', ec='blue', alpha=0.5)
    plt.title(f"Beta Radiation Deflection in {mat}\nThickness = {t_sim_beta:.3f} cm, σ = {sigma_beta:.3f} rad")
    plt.xlabel("Propagation distance (cm)")
    plt.ylabel("Lateral displacement (cm)")
    plt.xlim(0, 25)
    plt.ylim(-max(np.abs(offsets_beta))-2, max(np.abs(offsets_beta))+2)
    plt.grid(True)
    plt.show()
    
    # --- For Gamma Radiation ---
    mu_gamma = materials[mat]['gamma']
    t_req_gamma = 4.60517 / mu_gamma
    t_sim_gamma = t_req_gamma / 2  # representative thickness (cm)
    k_gamma = 0.05  # gamma rays are less deflected
    angles_gamma, offsets_gamma, sigma_gamma = simulate_deflection(t_sim_gamma, num_rays=100, k=k_gamma)
    
    plt.figure(figsize=(6, 4))
    for off in offsets_gamma:
        plt.arrow(0, 0, 20, off, head_width=0.5, head_length=0.5, fc='red', ec='red', alpha=0.5)
    plt.title(f"Gamma Radiation Deflection in {mat}\nThickness = {t_sim_gamma:.3f} cm, σ = {sigma_gamma:.3f} rad")
    plt.xlabel("Propagation distance (cm)")
    plt.ylabel("Lateral displacement (cm)")
    plt.xlim(0, 25)
    plt.ylim(-max(np.abs(offsets_gamma))-2, max(np.abs(offsets_gamma))+2)
    plt.grid(True)
    plt.show()
    
# ============================================================
# Finally, we also print the calculated amount of lead required for each 
# radiation type.
# (Focusing on Lead material from our dictionary.)
# ============================================================
print("=== Amount of Lead Required (for 99% attenuation) ===")
for rad in radiation_types:
    mu_lead = materials["Lead"][rad]
    t_req_lead = 4.60517 / mu_lead
    print(f"Radiation {rad:10s}: Required lead thickness = {t_req_lead:.4f} cm")
