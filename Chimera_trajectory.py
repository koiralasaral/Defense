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
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 2)  # Fit a 2nd degree polynomial
            trajectory_equation = f"y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
        else:
            trajectory_equation = "Insufficient data for polynomial fit"
            coeffs = [0, 0, 0]
        max_altitude = np.max(y)
        inclination_angle_deg = np.degrees(np.arctan2(vy,vx))


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
    return trajectory_equation, max_altitude, inclination_angle_deg, x, y

# Projectile motion analysis for each rocket and print trajectory equation
print("Projectile Motion Trajectory Equations:")
for rocket in ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']:
    equation, max_altitude, inclination_angle_deg, x_traj, y_traj = projectile_motion(rocket, atmosphere_model='spherical')
    print(f"{rocket}: {equation}")
    print(f"  Max Altitude: {max_altitude:.2f} km")
    print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")
    if len(x_traj) > 0:
        final_range = x_traj[-1]
        print(f"  Calculated Range: {final_range:.2f} km, Target Range: {rocket_data[rocket]['range_km']} km")

# =============================================================================
# 3. Range and Blast Energy Interpretation
# =============================================================================

print("\nRange and Blast Energy Interpretation:")
for rocket in rocket_data:
    print(f"{rocket}:")
    print(f"  Range is {rocket_data[rocket]['range_km']} km, indicating its reach.")
    print(f"  Blast energy is {rocket_data[rocket]['energy_joules']:.2e} Joules, representing its destructive potential.")

import re

# ... (rest of your code)

# =============================================================================
# 4. Interpretation of Trajectory Equations
# =============================================================================
print("\nInterpretation of Trajectory Equations:")
rockets = ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']
for target_rocket in rockets:
    print(f"\nTrajectory equation for {target_rocket}:")
    equation, max_altitude, inclination_angle_deg, x_traj, y_traj = projectile_motion(target_rocket, atmosphere_model='spherical')
    print(f"{target_rocket}: {equation}")
    print(f"  Max Altitude: {max_altitude:.2f} km")
    print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")
    if len(x_traj) > 0:
        final_range = x_traj[-1]
        print(f"  Calculated Range: {final_range:.2f} km, Target Range: {rocket_data[target_rocket]['range_km']} km")

    a_coeff = 0
    b_coeff = 0
    c_coeff = 0

    if "Insufficient data" not in equation:
        # Use regular expressions to find the coefficients
        match_a = re.search(r"([-+]?\d*\.?\d*)x\^2", equation)
        if match_a:
            a_coeff_str = match_a.group(1)
            if a_coeff_str in ('', '+'):
                a_coeff = 1.0
            elif a_coeff_str == '-':
                a_coeff = -1.0
            else:
                a_coeff = float(a_coeff_str)

        match_b = re.search(r"([-+]?\d*\.?\d*)x(?!\^)", equation) # Avoid matching x^2
        if match_b:
            b_coeff_str = match_b.group(1)
            if b_coeff_str in ('', '+'):
                b_coeff = 1.0
            elif b_coeff_str == '-':
                b_coeff = -1.0
            else:
                b_coeff = float(b_coeff_str)

        match_c = re.search(r"([-+]?\d*\.?\d+)(?!x)", equation) # Find the constant term
        if match_c:
            c_coeff = float(match_c.group(1))

    print(f"  This trajectory can be approximated by a parabola. The coefficients are:")
    print(f"    a (quadratic term): {a_coeff:.2f}, affecting the curvature.")
    print(f"    b (linear term): {b_coeff:.2f}, affecting the initial slope.")
    print(f"    c (constant term): {c_coeff:.2f}, representing the initial altitude (0 in this case).")

    # Compare with other rockets (simplified - just comparing coefficients)
    print("  Comparison with other rockets (simplified):")
    for other_rocket in rockets:
        if other_rocket != target_rocket:
            other_equation, other_max_altitude, other_inclination_angle_deg, other_x_traj, other_y_traj = projectile_motion(other_rocket, atmosphere_model='spherical')
            other_a_coeff = 0
            other_b_coeff = 0
            other_c_coeff = 0
            if "Insufficient data" not in other_equation:
                match_other_a = re.search(r"([-+]?\d*\.?\d*)x\^2", other_equation)
                if match_other_a:
                    other_a_coeff_str = match_other_a.group(1)
                    if other_a_coeff_str in ('', '+'):
                        other_a_coeff = 1.0
                    elif other_a_coeff_str == '-':
                        other_a_coeff = -1.0
                    else:
                        other_a_coeff = float(other_a_coeff_str)

                match_other_b = re.search(r"([-+]?\d*\.?\d*)x(?!\^)", other_equation)
                if match_other_b:
                    other_b_coeff_str = match_other_b.group(1)
                    if other_b_coeff_str in ('', '+'):
                        other_b_coeff = 1.0
                    elif other_b_coeff_str == '-':
                        other_b_coeff = -1.0
                    else:
                        other_b_coeff = float(other_b_coeff_str)

                match_other_c = re.search(r"([-+]?\d*\.?\d+)(?!x)", other_equation)
                if match_other_c:
                    other_c_coeff = float(match_other_c.group(1))

            def safe_division(num, den):
                if den == 0:
                    return "infinity" if num != 0 else "undefined"
                return f"{num / den:.2f}"

            print(f"    {other_rocket}: a is {safe_division(a_coeff, other_a_coeff)} times its a, b is {safe_division(b_coeff, other_b_coeff)} times its b. c is {safe_division(c_coeff, other_c_coeff)} times its c")

# ... (rest of your code)import re

# ... (rest of your code)

# =============================================================================
# 4. Interpretation of Trajectory Equations
# =============================================================================
print("\nInterpretation of Trajectory Equations:")
rockets = ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']
for target_rocket in rockets:
    print(f"\nTrajectory equation for {target_rocket}:")
    equation, max_altitude, inclination_angle_deg, x_traj, y_traj = projectile_motion(target_rocket, atmosphere_model='spherical')
    print(f"{target_rocket}: {equation}")
    print(f"  Max Altitude: {max_altitude:.2f} km")
    print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")
    if len(x_traj) > 0:
        final_range = x_traj[-1]
        print(f"  Calculated Range: {final_range:.2f} km, Target Range: {rocket_data[target_rocket]['range_km']} km")

    a_coeff = 0
    b_coeff = 0
    c_coeff = 0

    if "Insufficient data" not in equation:
        # Use regular expressions to find the coefficients
        match_a = re.search(r"([-+]?\d*\.?\d*)x\^2", equation)
        if match_a:
            a_coeff_str = match_a.group(1)
            if a_coeff_str in ('', '+'):
                a_coeff = 1.0
            elif a_coeff_str == '-':
                a_coeff = -1.0
            else:
                a_coeff = float(a_coeff_str)

        match_b = re.search(r"([-+]?\d*\.?\d*)x(?!\^)", equation) # Avoid matching x^2
        if match_b:
            b_coeff_str = match_b.group(1)
            if b_coeff_str in ('', '+'):
                b_coeff = 1.0
            elif b_coeff_str == '-':
                b_coeff = -1.0
            else:
                b_coeff = float(b_coeff_str)

        match_c = re.search(r"([-+]?\d*\.?\d+)(?!x)", equation) # Find the constant term
        if match_c:
            c_coeff = float(match_c.group(1))

    print(f"  This trajectory can be approximated by a parabola. The coefficients are:")
    print(f"    a (quadratic term): {a_coeff:.2f}, affecting the curvature.")
    print(f"    b (linear term): {b_coeff:.2f}, affecting the initial slope.")
    print(f"    c (constant term): {c_coeff:.2f}, representing the initial altitude (0 in this case).")

    # Compare with other rockets (simplified - just comparing coefficients)
    print("  Comparison with other rockets (simplified):")
    for other_rocket in rockets:
        if other_rocket != target_rocket:
            other_equation, other_max_altitude, other_inclination_angle_deg, other_x_traj, other_y_traj = projectile_motion(other_rocket, atmosphere_model='spherical')
            other_a_coeff = 0
            other_b_coeff = 0
            other_c_coeff = 0
            if "Insufficient data" not in other_equation:
                match_other_a = re.search(r"([-+]?\d*\.?\d*)x\^2", other_equation)
                if match_other_a:
                    other_a_coeff_str = match_other_a.group(1)
                    if other_a_coeff_str in ('', '+'):
                        other_a_coeff = 1.0
                    elif other_a_coeff_str == '-':
                        other_a_coeff = -1.0
                    else:
                        other_a_coeff = float(other_a_coeff_str)

                match_other_b = re.search(r"([-+]?\d*\.?\d*)x(?!\^)", other_equation)
                if match_other_b:
                    other_b_coeff_str = match_other_b.group(1)
                    if other_b_coeff_str in ('', '+'):
                        other_b_coeff = 1.0
                    elif other_b_coeff_str == '-':
                        other_b_coeff = -1.0
                    else:
                        other_b_coeff = float(other_b_coeff_str)

                match_other_c = re.search(r"([-+]?\d*\.?\d+)(?!x)", other_equation)
                if match_other_c:
                    other_c_coeff = float(match_other_c.group(1))

            def safe_division(num, den):
                if den == 0:
                    return "infinity" if num != 0 else "undefined"
                return f"{num / den:.2f}"

            print(f"    {other_rocket}: a is {safe_division(a_coeff, other_a_coeff)} times its a, b is {safe_division(b_coeff, other_b_coeff)} times its b. c is {safe_division(c_coeff, other_c_coeff)} times its c")

# ... (rest of your code)
# =============================================================================
# 5. Creating a New Rocket Based on Linear Combination for Maximum Energy
# =============================================================================

print("\nCreating a New Rocket for Maximum Energy:")

# Define the properties we'll use for the linear combination
properties = ['cost_million_usd', 'range_km', 'energy_joules']
num_rockets = len(rockets)

# Create the matrix A and vector b
A = np.zeros((num_rockets, num_rockets))
b = np.zeros(num_rockets)

# Fill matrix A and vector b
for i, rocket_name in enumerate(rockets):
    rocket = rocket_data[rocket_name]
    A[i] = [rocket[prop] for prop in properties]
    b[i] = rocket['energy_joules']  # Maximize energy

print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
# Solve the system of equations Ax = b to find the coefficients
try:
    x = np.linalg.solve(A, b)
    print("\nCoefficients x (for each rocket in linear combination):")
    print(x)
except np.linalg.LinAlgError:
    print("\nSingular matrix. Cannot solve for a unique solution. Using pseudoinverse instead")
    x = np.linalg.pinv(A) @ b
    print("\nCoefficients x (for each rocket in linear combination):")
    print(x)
# Ensure no negative coefficients
x = np.maximum(x, 0)

# Normalize the coefficients
x = x / np.sum(x)
print("Normalized coefficients")
print(x)
# Calculate properties of the new rocket
new_rocket_data = {}
for i, prop in enumerate(properties):
    new_rocket_data[prop] = sum(x[j] * rocket_data[rockets[j]][prop] for j in range(num_rockets))

new_rocket_name = "Optimus"  # Naming the new rocket
print(f"\nNew Rocket ({new_rocket_name}) Properties:")
for prop, value in new_rocket_data.items():
    print(f"  {prop}: {value:.2f}")

# Add the new rocket to the rocket_data dictionary
rocket_data[new_rocket_name] = {
    "range_km": new_rocket_data['range_km'],
    "tnt_equivalent_kg": new_rocket_data['energy_joules'] / 4.184e6,  # Convert back to kg
    "cd": 0.25,  # Assume a value, needs proper calculation
    "speed_km_s": 5, #Assume a value
    "length_m": 15, #Assume a value
    "diameter_m": 2.5, #Assume a value
    "type": "Composite",  # Mark it as a composite
    "cost_million_usd": new_rocket_data['cost_million_usd'],
    "energy_joules": new_rocket_data['energy_joules'],
}
# =============================================================================
# 6. MCMC Simulation for Blast Damage Area with New Rocket
# =============================================================================
target_area = 20000  # km^2

# Simplified crater radius approximation from energy.
def calculate_crater_radius_from_energy(energy_joules, material_strength=1e7):
    radius_m = (energy_joules / material_strength)**(1/3)
    return radius_m / 1000  # Convert to km

# Calculate blast areas for V2, Iskander, Minuteman_III, and Optimus
v2_area = np.pi * (calculate_crater_radius_from_energy(rocket_data['V2']['energy_joules']))**2
iskander_area = np.pi * (calculate_crater_radius_from_energy(rocket_data['Iskander']['energy_joules']))**2
minuteman_area = np.pi * (calculate_crater_radius_from_energy(rocket_data['Minuteman_III']['energy_joules']))**2
optimus_area = np.pi * (calculate_crater_radius_from_energy(rocket_data[new_rocket_name]['energy_joules']))**2

# Print the individual blast areas
print(f"\nBlast Damage Areas:")
print(f"V2: {v2_area:.2f} km^2")
print(f"Iskander: {iskander_area:.2f} km^2")
print(f"Minuteman_III: {minuteman_area:.2f} km^2")
print(f"{new_rocket_name}: {optimus_area:.2f} km^2")

# Define the variables using sympy
x_v2, x_iskander, x_minuteman, x_optimus = sp.symbols('x_v2 x_iskander x_minuteman x_optimus')

# Define the equation
equation = sp.Eq(v2_area*x_v2 + iskander_area*x_iskander + minuteman_area*x_minuteman + optimus_area*x_optimus, target_area)
print(f"\nBlast Damage Area Equation: {equation}")

# Case I: Using only the new rocket (Optimus)
x_optimus_solution = target_area / optimus_area
print(f"\nNumber of {new_rocket_name} required: {x_optimus_solution:.2f}")

# Case II:  Solving for a combination of rockets.  This requires constraints or an optimization method.
# For demonstration, let's assume a constraint: x_v2 + x_iskander + x_minuteman + x_optimus = 1 (Not realistic, just for illustration)
solution = sp.solve([equation, x_v2 + x_iskander + x_minuteman + x_optimus - 1], (x_v2, x_iskander, x_minuteman, x_optimus))
print("\nSolution for combination of rockets (Illustrative, assumes sum = 1):")
print(solution)

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



# Example parabola
plot_blast_area_parabola([0.1, 0.5, 10], "Blast Area Parabola (Example)")

# Plot Chimera's Trajectory
equation, max_altitude, inclination_angle_deg = projectile_motion(new_rocket_name, atmosphere_model='spherical')
print(f"\n{new_rocket_name} Trajectory Equation: {equation}")
print(f"  Max Altitude: {max_altitude:.2f} km")
print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")