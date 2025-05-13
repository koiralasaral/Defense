import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# 1. Data Collection and Preparation
# =============================================================================

# Rocket Data (using more accurate real-world values)
rocket_data = {
    "V2": {
        "range_km": 320,
        "tnt_equivalent_kg": 1000,
        "cd": 0.15,
        "speed_km_s": 1.6,
        "length_m": 14,
        "diameter_m": 1.65,
        "type": "Ballistic",
        "cost_million_usd": 0.3, #estimated
    },
    "Iskander": {
        "range_km": 500,
        "tnt_equivalent_kg": 480,
        "cd": 0.30,
        "speed_km_s": 2.1,
        "length_m": 7.3,
        "diameter_m": 0.92,
        "type": "Ballistic",
        "cost_million_usd": 3
    },
    "Tomahawk": {
        "range_km": 2500,
        "tnt_equivalent_kg": 454,
        "cd": 0.30,
        "speed_km_s": 0.24,
        "length_m": 6.25,
        "diameter_m": 0.52,
        "type": "Cruise",
        "cost_million_usd": 1.5
    },
    "Minuteman_III": {
        "range_km": 13000,
        "tnt_equivalent_kg": 300000,  # 300 kt
        "cd": 0.20,
        "speed_km_s": 7,
        "length_m": 18.2,
        "diameter_m": 1.67,
        "type": "ICBM",
        "cost_million_usd": 7
    },
    "Topol_M": {
        "range_km": 11000,
        "tnt_equivalent_kg": 1000000, # 1MT
        "cd": 0.22,
        "speed_km_s": 7.3,
        "length_m": 22.7,
        "diameter_m": 1.86,
        "type": "ICBM",
        "cost_million_usd": 8
    },
    "DF_41": {
        "range_km": 14000,
        "tnt_equivalent_kg": 1000000, # 1MT
        "cd": 0.21,
        "speed_km_s": 7.8,
        "length_m": 16.5,
        "diameter_m": 2,
        "type": "ICBM",
        "cost_million_usd": 10
    },
    "Sarmat":{
        "range_km": 18000,
        "tnt_equivalent_kg": 8000000, # 8MT
        "cd": 0.23,
        "speed_km_s": 7.5,
        "length_m": 35.5,
        "diameter_m": 3,
        "type": "ICBM",
        "cost_million_usd": 20
    }
}

# Convert TNT equivalent to Joules
for rocket in rocket_data:
    rocket_data[rocket]["energy_joules"] = rocket_data[rocket]["tnt_equivalent_kg"] * 4.184e6

# =============================================================================
# 2. Projectile Motion Analysis
# =============================================================================
def projectile_motion(rocket_name, launch_angle_deg=45, atmosphere_model='flat'):
    """
    Calculates and plots the projectile motion of a rocket, and returns the equation of the trajectory.

    Args:
        rocket_name (str): Name of the rocket.
        launch_angle_deg (float): Launch angle in degrees.
        atmosphere_model (str): 'flat' or 'spherical'.

    Returns:
        str: Equation of the trajectory.
    """
    rocket = rocket_data[rocket_name]
    range_km = rocket["range_km"]
    speed_km_s = rocket["speed_km_s"]
    cd = rocket["cd"]
    launch_angle_rad = np.radians(launch_angle_deg)
    g = 9.81 / 1000  # Convert to km/s^2

    if atmosphere_model == 'flat':
        # Simplified flat-Earth trajectory calculation (no drag)
        t_flight = 2 * speed_km_s * np.sin(launch_angle_rad) / g
        time_points = np.linspace(0, t_flight, 100)
        x = speed_km_s * np.cos(launch_angle_rad) * time_points
        y = speed_km_s * np.sin(launch_angle_rad) * time_points - 0.5 * g * time_points**2
        trajectory_equation = f"y = {speed_km_s * np.tan(launch_angle_rad):.2f}x - {0.5 * g / (speed_km_s * np.cos(launch_angle_rad))**2:.2f}x^2"


    elif atmosphere_model == 'spherical':
        R = 6371  # Earth radius in km
        # More complex, iterative calculation with a simplified drag model.
        dt = 0.1  # Time step
        t = 0
        x = 0
        y = 0
        vx = speed_km_s * np.cos(launch_angle_rad)
        vy = speed_km_s * np.sin(launch_angle_rad)
        trajectory_x = [x]
        trajectory_y = [y]
        while y >= 0:
            t += dt
            # Simplified drag force (proportional to velocity squared)
            drag_x = -0.5 * cd * (vx**2 + vy**2) * vx  # Simplified drag
            drag_y = -0.5 * cd * (vx**2 + vy**2) * vy
            vx += (drag_x) * dt
            vy += (-g + drag_y) * dt
            x += vx * dt
            y += vy * dt
            trajectory_x.append(x)
            trajectory_y.append(y)
        time_points = np.linspace(0, t, len(trajectory_x))
        x = trajectory_x
        y = trajectory_y
        # Fit a polynomial to the trajectory data to get an approximate equation
        coeffs = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
        trajectory_equation = f"y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"

    else:
        raise ValueError("atmosphere_model must be 'flat' or 'spherical'")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"{rocket_name} Trajectory")
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (km)")
    plt.title(f"{rocket_name} Trajectory Simulation ({atmosphere_model} Atmosphere)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    return trajectory_equation

# Projectile motion analysis for each rocket and print trajectory equation
print("Projectile Motion Trajectory Equations:")
for rocket in ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']:
    equation = projectile_motion(rocket, atmosphere_model='spherical')
    print(f"{rocket}: {equation}")

# =============================================================================
# 3. Range and Blast Energy Interpretation
# =============================================================================

print("\nRange and Blast Energy Interpretation:")
for rocket in rocket_data:
    print(f"{rocket}:")
    print(f"  Range is {rocket_data[rocket]['range_km']} km, indicating its reach.")
    print(f"  Blast energy is {rocket_data[rocket]['energy_joules']:.2e} Joules, representing its destructive potential.")

# =============================================================================
# 4. Linear Combination for Blast Radius
# =============================================================================

# Target blast radius
target_blast_radius_km2 = 20000

# Available rockets and their properties
available_rockets = ['V2', 'Iskander', 'Minuteman_III']
rocket_costs = [rocket_data[r]['cost_million_usd'] for r in available_rockets]
rocket_energies = [rocket_data[r]['energy_joules'] for r in available_rockets]

#  Crater radius approximation (in km)
def calculate_crater_radius_km(energy_joules, material_strength=1e7):
    """ Simplified crater radius calculation (more conceptual than precise)"""
    radius_m = (energy_joules / material_strength)**(1/3)
    return radius_m / 1000 #radius in km

# Calculate blast radius for each rocket
rocket_blast_radii_km2 = [np.pi * (calculate_crater_radius_km(energy))**2 for energy in rocket_energies] #blast radius in km2

# Linear regression to find optimal combination
X = np.array([rocket_blast_radii_km2]).T  # Independent variables (blast radii)
y = np.array([target_blast_radius_km2])      # Dependent variable (target blast radius)

# Add a constant to the model
X = sm.add_constant(X)

# Case I: Minimize Cost (approximately)
# Weight cost inversely to blast radius contribution
weights_cost = [1 / cost for cost in rocket_costs]
weights_cost_normalized = weights_cost / np.sum(weights_cost) #normalizing the weights

# Create a design matrix W, where each column represents the weighted contribution of each rocket.
W_cost = np.diag(weights_cost_normalized)
# Solve for the coefficients using the normal equation
coefficients_cost = np.linalg.solve(X.T @ W_cost @ X, X.T @ W_cost @ y)
print(f"\nLinear Regression Equation (Minimum Cost):\nTarget Blast Radius = {coefficients_cost[0]:.2f} + {coefficients_cost[1]:.2f} * (V2 Blast Radius) + {coefficients_cost[2]:.2f} * (Iskander Blast Radius) + {coefficients_cost[3]:.2f} * (Minuteman_III Blast Radius)")


# Predicted blast radius
predicted_blast_radius_cost = X @ coefficients_cost

# Case II: Maximize Energy (approximately)
# Weight energy directly
weights_energy = [(energy) for energy in rocket_energies]
weights_energy_normalized = weights_energy / np.sum(weights_energy)

# Create a design matrix W, where each column represents the weighted contribution of each rocket.
W_energy = np.diag(weights_energy_normalized)
# Solve for the coefficients using the normal equation
coefficients_energy = np.linalg.solve(X.T @ W_energy @ X, X.T @ W_energy @ y)
print(f"\nLinear Regression Equation (Maximum Energy):\nTarget Blast Radius = {coefficients_energy[0]:.2f} + {coefficients_energy[1]:.2f} * (V2 Blast Radius) + {coefficients_energy[2]:.2f} * (Iskander Blast Radius) + {coefficients_energy[3]:.2f} * (Minuteman_III Blast Radius)")
predicted_blast_radius_energy = X @ coefficients_energy

# =============================================================================
# 5. Parabola Visualization
# =============================================================================

# Generate x values for the parabola
x_values = np.linspace(0, 25000, 1000)  # Adjust range as needed

# Calculate y values using the linear regression equations
y_cost = coefficients_cost[0] + coefficients_cost[1] * (np.pi * (x_values/1000)**2)  # Simplified parabola equation
y_energy = coefficients_energy[0] + coefficients_energy[1] * (np.pi * (x_values/1000)**2)

# Plot the parabolas
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_cost, label="Minimum Cost Parabola")
plt.plot(x_values, y_energy, label="Maximum Energy Parabola")
plt.xlabel("Distance (km)")
plt.ylabel("Blast Radius (km^2)")
plt.title("Parabola Visualization of Blast Radius")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
