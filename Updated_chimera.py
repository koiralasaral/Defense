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
import pymc as pm
import arviz as az
import sympy as sp

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
    g = 9.81 / 1000  # Convert to km/s^2

    if atmosphere_model == 'flat':
        # Calculate the launch angle required to achieve the given range (flat earth, no drag)
        if speed_km_s > 0:
            launch_angle_rad = 0.5 * np.arcsin(g * range_km / (speed_km_s**2))
            launch_angle_deg = np.degrees(launch_angle_rad)
        else:
            launch_angle_rad = np.radians(launch_angle_deg) #if speed is zero, use default
        t_flight = 2 * speed_km_s * np.sin(launch_angle_rad) / g
        time_points = np.linspace(0, t_flight, 100)
        x = speed_km_s * np.cos(launch_angle_rad) * time_points
        y = speed_km_s * np.sin(launch_angle_rad) * time_points - 0.5 * g * time_points**2
        trajectory_equation = f"y = {speed_km_s * np.tan(launch_angle_rad):.2f}x - {0.5 * g / (speed_km_s * np.cos(launch_angle_rad))**2:.2f}x^2"
        max_altitude = (speed_km_s * np.sin(launch_angle_rad))**2 / (2 * g)
        inclination_angle_deg = launch_angle_deg


    elif atmosphere_model == 'spherical':
        R = 6371  # Earth radius in km
        # Iterative calculation with a simplified drag model.
        def calculate_trajectory(angle_deg):
            angle_rad = np.radians(angle_deg)
            t = 0
            x = 0
            y = 0
            vx = speed_km_s * np.cos(angle_rad)
            vy = speed_km_s * np.sin(angle_rad)
            trajectory_x = [x]
            trajectory_y = [y]
            dt = 0.1
            while y >= 0:
                t += dt
                drag_x = -0.5 * cd * (vx**2 + vy**2) * vx
                drag_y = -0.5 * cd * (vx**2 + vy**2) * vy
                vx += (drag_x) * dt
                vy += (-g + drag_y) * dt
                x += vx * dt
                y += vy * dt
                trajectory_x.append(x)
                trajectory_y.append(y)
            return trajectory_x, trajectory_y, t, vx, vy # Return t, vx, and vy

        # Use a binary search or optimization to find the launch angle that achieves the desired range
        lower_angle = 0
        upper_angle = 90
        max_iterations = 20
        tolerance = 1  # km
        vx = 0
        vy = 0

        for _ in range(max_iterations):
            mid_angle = (lower_angle + upper_angle) / 2
            trajectory_x, trajectory_y, t, vx, vy = calculate_trajectory(mid_angle) # Capture t, vx, and vy
            actual_range = trajectory_x[-1]  # Final x value

            if abs(actual_range - range_km) < tolerance:
                break
            elif actual_range < range_km:
                lower_angle = mid_angle
            else:
                upper_angle = mid_angle
        launch_angle_deg = mid_angle
        launch_angle_rad = np.radians(launch_angle_deg)
        x = trajectory_x
        y = trajectory_y
        time_points = np.linspace(0, t, len(trajectory_x))

        # Fit a polynomial to the trajectory data to get an approximate equation
        coeffs = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
        trajectory_equation = f"y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
        max_altitude = np.max(y)
        inclination_angle_deg = np.degrees(np.arctan(vy/vx))


    else:
        raise ValueError("atmosphere_model must be 'flat' or 'spherical'")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=f"{rocket_name} Trajectory")
    plt.xlabel("Distance (km)")
    plt.ylabel("Altitude (km)")
    plt.title(f"{rocket_name} Trajectory Simulation ({atmosphere_model} Atmosphere)\nRange: {range_km} km, Launch Angle: {launch_angle_deg:.2f} deg") # Added launch angle
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.xlim(0, range_km)  # Set x-axis limit to the rocket's range
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.show()
    return trajectory_equation, max_altitude, inclination_angle_deg

# Projectile motion analysis for each rocket and print trajectory equation
print("Projectile Motion Trajectory Equations:")
for rocket in ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']:
    equation, max_altitude, inclination_angle_deg = projectile_motion(rocket, atmosphere_model='spherical')
    print(f"{rocket}: {equation}")
    print(f"   Max Altitude: {max_altitude:.2f} km")
    print(f"   Initial Inclination Angle: {inclination_angle_deg:.2f} deg")

# =============================================================================
# 3. Range and Blast Energy Interpretation
# =============================================================================

print("\nRange and Blast Energy Interpretation:")
for rocket in rocket_data:
    print(f"{rocket}:")
    print(f"  Range is {rocket_data[rocket]['range_km']} km, indicating its reach.")
    print(f"  Blast energy is {rocket_data[rocket]['energy_joules']:.2e} Joules, representing its destructive potential.")

# =============================================================================
# 4. Interpretation of Trajectory Equations
# =============================================================================
print("\nInterpretation of Trajectory Equations:")
rockets = ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']
for target_rocket in rockets:
    print(f"\nTrajectory equation for {target_rocket}:")
    equation, max_altitude, inclination_angle_deg = projectile_motion(target_rocket, atmosphere_model='spherical')
    print(f"{target_rocket}: {equation}")
    print(f"   Max Altitude: {max_altitude:.2f} km")
    print(f"   Initial Inclination Angle: {inclination_angle_deg:.2f} deg")
    # Extract coefficients
    coeffs = equation.replace("y = ", "").split("x")
    if len(coeffs) > 1:
        a_coeff = float(coeffs[0].split("x^2 + ")[0])
        b_coeff_str = coeffs[1].split(" + ")[0].strip()  # Remove extra spaces
        b_coeff = float(b_coeff_str)
        c_coeff = float(coeffs[1].split(" + ")[1])
    else:
        a_coeff = 0
        b_coeff = 0
        c_coeff = float(coeffs[0])

    print(f"  This trajectory can be approximated by a parabola. The coefficients are:")
    print(f"    a (quadratic term): {a_coeff:.2f}, affecting the curvature.")
    print(f"    b (linear term): {b_coeff:.2f}, affecting the initial slope.")
    print(f"    c (constant term): {c_coeff:.2f}, representing the initial altitude (0 in this case).")

    # Compare with other rockets (simplified - just comparing coefficients)
    print("  Comparison with other rockets (simplified):")
    for other_rocket in rockets:
        if other_rocket != target_rocket:
            other_equation, other_max_altitude, other_inclination_angle_deg = projectile_motion(other_rocket, atmosphere_model='spherical')
            other_coeffs = other_equation.replace("y = ", "").split("x")
            if len(other_coeffs) > 1:
                other_a_coeff = float(other_coeffs[0].split("x^2 + ")[0])
                other_b_coeff_str = other_coeffs[1].split(" + ")[0].strip()
                other_b_coeff = float(other_b_coeff_str)
                other_c_coeff = float(other_coeffs[1].split(" + ")[1])
            else:
                other_a_coeff = 0
                other_b_coeff = 0
                other_c_coeff = float(other_coeffs[0])
            print(f"    {other_rocket}: a is {a_coeff/other_a_coeff:.2f} times its a, b is {b_coeff/other_b_coeff:.2f} times its b.  c is {c_coeff/other_c_coeff:.2f} times its c")

# =============================================================================
# 5. MCMC Simulation for Theoretical Missile
# =============================================================================

# 5. MCMC Simulation for Theoretical Missile
num_samples = 1000
# Priors (mean and standard deviation for each parameter)
prior_range_mean = np.mean([rocket_data[r]['range_km'] for r in rockets])
prior_range_std = np.std([rocket_data[r]['range_km'] for r in rockets])
prior_energy_mean = np.mean([rocket_data[r]['energy_joules'] for r in rockets])
prior_energy_std = np.std([rocket_data[r]['energy_joules'] for r in rockets])
prior_cost_mean = np.mean([rocket_data[r]['cost_million_usd'] for r in rockets])
prior_cost_std = np.std([rocket_data[r]['cost_million_usd'] for r in rockets])


# MCMC model
with pm.Model() as theoretical_missile_model:
    # Priors
    combined_range = pm.Normal('combined_range', mu=prior_range_mean, sigma=prior_range_std)
    combined_energy = pm.Normal('combined_energy', mu=prior_energy_mean, sigma=prior_energy_std)
    combined_cost = pm.Normal('combined_cost', mu=prior_cost_mean, sigma=prior_cost_std)

    # Define likelihood (simplified - assuming a normal distribution for the combined parameters)
    #  Very simplified likelihood, just to get *some* posterior.  This is NOT a physically realistic model.
    likelihood_range = pm.Normal('likelihood_range', mu=combined_range, sigma=prior_range_std, observed=np.sum([rocket_data[r]['range_km'] for r in rockets]))
    likelihood_energy = pm.Normal('likelihood_energy', mu=combined_energy, sigma=prior_energy_std, observed=np.sum([rocket_data[r]['energy_joules'] for r in rockets]))
    likelihood_cost = pm.Normal('likelihood_cost', mu=combined_cost, sigma=prior_cost_std, observed=np.sum([rocket_data[r]['cost_million_usd'] for r in rockets]))


    # Inference
    theoretical_missile_trace = pm.sample(num_samples, tune=1000, chains=2)

# Print mean and std of the combined parameters.
print("\nTheoretical Missile Parameters (from MCMC):")
print(f"Combined Range: {az.summary(theoretical_missile_trace, var_names=['combined_range']).mean['combined_range']:.2f} km (± {az.summary(theoretical_missile_trace, var_names=['combined_range']).sd['combined_range']:.2f})")
print(f"Combined Energy: {az.summary(theoretical_missile_trace, var_names=['combined_energy']).mean['combined_energy']:.2e} Joules (± {az.summary(theoretical_missile_trace, var_names=['combined_energy']).sd['combined_energy']:.2e})")
print(f"Combined Cost: {az.summary(theoretical_missile_trace, var_names=['combined_cost']).mean['combined_cost']:.2f} Million USD (± {az.summary(theoretical_missile_trace, var_names=['combined_cost']).sd['combined_cost']:.2f})")

# Name the theoretical missile
theoretical_missile_name = "Chimera"
print(f"\nTheoretical Missile Name: {theoretical_missile_name}")

# Extract Chimera's stats
chimera_range = az.summary(theoretical_missile_trace, var_names=['combined_range']).mean['combined_range']
chimera_energy = az.summary(theoretical_missile_trace, var_names=['combined_energy']).mean['combined_energy']
chimera_cost = az.summary(theoretical_missile_trace, var_names=['combined_cost']).mean['combined_cost']
print(f"\n{theoretical_missile_name} Stats:")
print(f"  Range: {chimera_range:.2f} km")
print(f"  Energy: {chimera_energy:.2e} Joules")
print(f"  Cost: {chimera_cost:.2f} Million USD")

# =============================================================================
# 6. Blast Damage Area Calculation
# =============================================================================

# Define the variables using sympy
x, y, z = sp.symbols('x y z')  #  x, y, z  represent the weapons.

# Target blast damage area
target_area = 20000  # km^2

#  Simplified crater radius approximation from energy.
def calculate_crater_radius_from_energy(energy_joules, material_strength=1e7):
    radius_m = (energy_joules / material_strength)**(1/3)
    return radius_m / 1000  # Convert to km

# Get energy for each weapon
v2_energy = rocket_data['V2']['energy_joules']
iskander_energy = rocket_data['Iskander']['energy_joules']
minuteman_energy = rocket_data['Minuteman_III']['energy_joules']

# Calculate blast areas.
v2_area = np.pi * (calculate_crater_radius_from_energy(v2_energy))**2
iskander_area = np.pi * (calculate_crater_radius_from_energy(iskander_energy))**2
minuteman_area = np.pi * (calculate_crater_radius_from_energy(minuteman_energy))**2

# Define the equation
equation = sp.Eq(v2_area*x + iskander_area*y + minuteman_area*z, target_area)
print(f"\nBlast Damage Area Equation: {equation}")

# Case I: Minimize Cost
# Cost coefficients
v2_cost = rocket_data['V2']['cost_million_usd']
iskander_cost = rocket_data['Iskander']['cost_million_usd']
minuteman_cost = rocket_data['Minuteman_III']['cost_million_usd']

# Objective: Minimize v2_cost*x + iskander_cost*y + minuteman_cost*z
# Use sympy to express the cost
cost_expression = v2_cost*x + iskander_cost*y + minuteman_cost*z

# Since this is a linear equation with 3 variables, there are infinite solutions.
#  Need an optimization library or constraints to find a *specific* solution that minimizes cost.
#  For simplicity, let's assume a constraint that x+y+z = 1 (i.e., we want a combination of 3 weapons to achieve the target area)
#  This is NOT a realistic assumption, but it allows us to demonstrate the concept.

#  Solve using sympy (very simplified example)
solution_cost = sp.solve([equation, x + y + z - 1], (x, y, z)) # Find a solution
print("\nMinimum Cost Solution (Illustrative Example, assumes x+y+z=1):", solution_cost)

# Case II: Maximize Energy
# Energy coefficients
v2_energy_val = rocket_data['V2']['energy_joules']
iskander_energy_val = rocket_data['Iskander']['energy_joules']
minuteman_energy_val = rocket_data['Minuteman_III']['energy_joules']

# Objective: Maximize v2_energy_val*x + iskander_energy_val*y + minuteman_energy_val*z
energy_expression = v2_energy_val*x + iskander_energy_val*y + minuteman_energy_val*z
solution_energy = sp.solve([equation, x + y + z - 1], (x, y, z))
print("Maximum Energy Solution (Illustrative Example, assumes x+y+z=1):", solution_energy)

# =============================================================================
# 7. Parabola Visualization for Blast Damage Area
# =============================================================================
def plot_blast_area_parabola(coefficients, title):
    """
    Plots a parabola representing the blast damage area based on a simplified quadratic equation.

    Args:
        coefficients (list): Coefficients [A, B, C] for the equation Area = A*distance^2 + B*distance + C.
        title (str): Title of the plot.
    """
    distance = np.linspace(0, 500, 1000)  # Distance in km (adjust as needed)
    area = coefficients[0] * distance**2 + coefficients[1] * distance + coefficients[2]

    plt.figure(figsize=(8, 6))
    plt.plot(distance, area)
    plt.xlabel("Distance (km)")
    plt.ylabel("Blast Damage Area (km^2)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  The solutions from sympy are points, not parabolas.  To *show* a parabola,
#  we need a quadratic equation relating blast area to some other parameter, like distance.
#  Let's make a simplified example:  Area = A*distance^2 + B*distance + C, and arbitrarily choose A,B,C
#  to plot.  In a real scenario, A,B,C would come from a physical model or regression.

# Example parabola for Minimum Cost scenario
plot_blast_area_parabola([0.1, 0.5, 10], "Blast Area Parabola (Minimum Cost Example)")

# Example parabola for Maximum Energy scenario
plot_blast_area_parabola([0.2, 0.8, 5], "Blast Area Parabola (Maximum Energy Example)")

# Plot Chimera's Trajectory
equation, max_altitude, inclination_angle_deg = projectile_motion(theoretical_missile_name, atmosphere_model='spherical')
print(f"\n{theoretical_missile_name} Trajectory Equation: {equation}")
print(f"  Max Altitude: {max_altitude:.2f} km")
print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")

