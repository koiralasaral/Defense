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
import sympy as sp
import re

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
        "energy_joules": 1000 * 4.184e6
    },
    "Iskander": {
        "range_km": 500,
        "tnt_equivalent_kg": 480,
        "cd": 0.30,
        "speed_km_s": 2.1,
        "length_m": 7.3,
        "diameter_m": 0.92,
        "type": "Ballistic",
        "cost_million_usd": 3,
        "energy_joules": 480 * 4.184e6
    },
    "Tomahawk": {
        "range_km": 2500,
        "tnt_equivalent_kg": 454,
        "cd": 0.30,
        "speed_km_s": 0.24,
        "length_m": 6.25,
        "diameter_m": 0.52,
        "type": "Cruise",
        "cost_million_usd": 1.5,
        "energy_joules": 454 * 4.184e6
    },
    "Minuteman_III": {
        "range_km": 13000,
        "tnt_equivalent_kg": 300000,   # 300 kt
        "cd": 0.20,
        "speed_km_s": 7,
        "length_m": 18.2,
        "diameter_m": 1.67,
        "type": "ICBM",
        "cost_million_usd": 7,
        "energy_joules": 300000 * 4.184e6
    },
    "Topol_M": {
        "range_km": 11000,
        "tnt_equivalent_kg": 1000000, # 1MT
        "cd": 0.22,
        "speed_km_s": 7.3,
        "length_m": 22.7,
        "diameter_m": 1.86,
        "type": "ICBM",
        "cost_million_usd": 8,
        "energy_joules": 1000000 * 4.184e6
    },
    "DF_41": {
        "range_km": 14000,
        "tnt_equivalent_kg": 1000000, # 1MT
        "cd": 0.21,
        "speed_km_s": 7.8,
        "length_m": 16.5,
        "diameter_m": 2,
        "type": "ICBM",
        "cost_million_usd": 10,
        "energy_joules": 1000000 * 4.184e6
    },
    "Sarmat":{
        "range_km": 18000,
        "tnt_equivalent_kg": 8000000, # 8MT
        "cd": 0.23,
        "speed_km_s": 7.5,
        "length_m": 35.5,
        "diameter_m": 3,
        "type": "ICBM",
        "cost_million_usd": 20,
        "energy_joules": 8000000 * 4.184e6
    }
}

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
        tuple: Equation of the trajectory (str), maximum altitude (float),
               initial inclination angle (float), x-trajectory (list), y-trajectory (list).
    """
    rocket = rocket_data[rocket_name]
    range_km = rocket["range_km"]
    speed_km_s = rocket["speed_km_s"]
    cd = rocket["cd"]
    g = 9.81 / 1000   # Convert to km/s^2

    if atmosphere_model == 'flat':
        # Calculate the launch angle required to achieve the given range (flat earth, no drag)
        if speed_km_s > 0:
            # Ensure the argument of arcsin is within [-1, 1]
            asin_arg = g * range_km / (speed_km_s**2)
            if -1 <= asin_arg <= 1:
                launch_angle_rad = 0.5 * np.arcsin(asin_arg)
                launch_angle_deg = np.degrees(launch_angle_rad)
            else:
                launch_angle_rad = np.radians(launch_angle_deg) # Use default if calculation fails
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
        R = 6371   # Earth radius in km
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
        tolerance = 1   # km
        vx = 0
        vy = 0
        mid_angle = launch_angle_deg # Initialize with a positive angle

        for _ in range(max_iterations):
            trajectory_x, trajectory_y, t, vx, vy = calculate_trajectory(mid_angle) # Capture t, vx, and vy
            actual_range = trajectory_x[-1]   # Final x value

            if abs(actual_range - range_km) < tolerance:
                break
            elif actual_range < range_km:
                lower_angle = mid_angle
            else:
                upper_angle = mid_angle
            mid_angle = (lower_angle + upper_angle) / 2

        launch_angle_deg = mid_angle
        launch_angle_rad = np.radians(launch_angle_deg)
        x = trajectory_x
        y = trajectory_y
        time_points = np.linspace(0, t, len(trajectory_x))

        # Fit a polynomial to the trajectory data to get an approximate equation
        if len(x) > 2:
            coeffs = np.polyfit(x, y, 2)   # Fit a 2nd degree polynomial
            trajectory_equation = f"y = {coeffs[0]:.2f}x^2 + {coeffs[1]:.2f}x + {coeffs[2]:.2f}"
        else:
            trajectory_equation = "Insufficient data for polynomial fit"
            coeffs = [0, 0, 0]
        max_altitude = np.max(y)
        inclination_angle_deg = launch_angle_deg # Corrected to use the calculated launch angle

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
    plt.xlim(0, range_km * 1.1)   # Set x-axis limit slightly beyond the range
    plt.ylim(bottom=0) # Ensure y-axis starts at 0
    plt.show()
    return trajectory_equation, max_altitude, inclination_angle_deg, np.array(x), np.array(y)

# Projectile motion analysis for each rocket and print trajectory equation
print("Projectile Motion Trajectory Equations:")
trajectory_data = {}
for rocket in ['V2', 'Iskander', 'Tomahawk', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']:
    equation, max_altitude, inclination_angle_deg, x_traj, y_traj = projectile_motion(rocket, launch_angle_deg=45, atmosphere_model='spherical') #set launch angle here
    trajectory_data[rocket] = {'equation': equation, 'max_altitude': max_altitude,
                                   'inclination_angle': inclination_angle_deg, 'x': x_traj, 'y': y_traj}
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

# =============================================================================
# 4. Interpretation of Trajectory Equations
# =============================================================================
print("\nInterpretation of Trajectory Equations:")
for target_rocket in trajectory_data: # Corrected line: iterating through the keys of trajectory_data
    print(f"\nTrajectory equation for {target_rocket}:")
    equation = trajectory_data[target_rocket]['equation']
    max_altitude = trajectory_data[target_rocket]['max_altitude']
    inclination_angle_deg = trajectory_data[target_rocket]['inclination_angle']
    x_traj = trajectory_data[target_rocket]['x']
    y_traj = trajectory_data[target_rocket]['y']

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
            c_coeff_str = match_c.group(1)
            if c_coeff_str in ('', '+'):
                c_coeff = 1.0
            elif c_coeff_str == '-':
                c_coeff = -1.0
            else:
                c_coeff = float(c_coeff_str)
        else:
            c_coeff = 0.0  # added this line

    print(f"  This trajectory can be approximated by a parabola. The coefficients are:")
    print(f"    a (quadratic term): {a_coeff:.2f}, affecting the curvature.")
    print(f"    b (linear term): {b_coeff:.2f}, affecting the initial slope.")
    print(f"    c (constant term): {c_coeff:.2f}, representing the initial altitude (0 in this case).")

    # Compare with other rocket (simplified - just comparing coefficients)
    print("  Comparison with other rocket (simplified):")
    for other_rocket in trajectory_data: # Corrected line: iterating through the keys of trajectory_data
        if other_rocket != target_rocket:
            other_equation = trajectory_data[other_rocket]['equation']
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
                else:
                    other_a_coeff = 0.0

                match_other_b = re.search(r"([-+]?\d*\.?\d*)x(?!\^)", other_equation)
                if match_other_b:
                    other_b_coeff_str = match_other_b.group(1)
                    if other_b_coeff_str in ('', '+'):
                        other_b_coeff = 1.0
                    elif other_b_coeff_str == '-':
                        other_b_coeff = -1.0
                    else:
                        other_b_coeff = float(other_b_coeff_str)
                else:
                    other_b_coeff = 0.0

                match_other_c = re.search(r"([-+]?\d*\.?\d+)(?!x)", other_equation)
                if match_other_c:
                    other_c_coeff_str = match_other_c.group(1)
                    if other_c_coeff_str in ('', '+'):
                        other_c_coeff = 1.0
                    elif other_c_coeff_str == '-':
                        other_c_coeff = -1.0
                    else:
                        other_c_coeff = float(other_c_coeff_str)
                else:
                    other_c_coeff = 0.0 #added this line

            def safe_division(num, den):
                if den == 0:
                    return "infinity" if num != 0 else "undefined"
                return f"{num / den:.2f}"

            print(f"    {other_rocket}: a is {safe_division(a_coeff, other_a_coeff)} times its a, b is {safe_division(b_coeff, other_b_coeff)} times its b. c is {safe_division(c_coeff, other_c_coeff)} times its c")
# =============================================================================
# 5. Creating a New Rocket Based on Linear Combination for Maximum Energy
# =============================================================================

print("\nCreating a New Rocket for Maximum Energy:")

# Define the properties we'll use for the linear combination
properties = ['cost_million_usd', 'range_km', 'energy_joules']
num_rocket = len(rocket_data) #changed rocket to rocket_name

# Create the matrix A and vector b
A = np.zeros((num_rocket, num_rocket))
b = np.zeros(num_rocket)

for i, rocket_name in enumerate(rocket_data.keys()): #changed rocket to rocket_name
    current_rocket = rocket_data[rocket_name]  # Use a new variable here
    A[i] = [current_rocket[prop] for prop in properties]
    b[i] = current_rocket['energy_joules']  # Maximize energy
# Normalize the matrix A
A = A / np.linalg.norm(A, axis=0)
# Normalize the vector b
b = b / np.linalg.norm(b)
# Print the normalized matrix A and vector b
print("\nNormalized Matrix A and Vector b:")
print("Normalized Matrix A:")
print(A)
print("\nNormalized Vector b:")
print(b)
# Print the original matrix A and vector b
print("\nOriginal Matrix A and Vector b:")
A = np.zeros((num_rocket, num_rocket))
b = np.zeros(num_rocket)
for i, rocket_name in enumerate(rocket_data.keys()):
    current_rocket = rocket_data[rocket_name]  # Use a new variable here
    A[i] = [current_rocket[prop] for prop in properties]
    b[i] = current_rocket['energy_joules']  # Maximize energy
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
    new_rocket_data[prop] = sum(x[j] * rocket_data[list(rocket_data.keys())[j]][prop] for j in range(num_rocket)) #changed rocket to rocket_data.keys()

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
print(f"\nNew Rocket ({new_rocket_name}) added to the data:")
for prop, value in rocket_data[new_rocket_name].items():
    print(f"  {prop}: {value:.2f}")
# =============================================================================
# =============================================================================
# 6. MCMC Simulation for Blast Damage Area with New Rocket
# =============================================================================
target_area = 20000   # km^2

# Simplified crater radius approximation from energy.
def calculate_crater_radius_from_energy(energy_joules, material_strength=1e7):
    radius_m = (energy_joules / material_strength)**(1/3)
    return radius_m / 1000   # Convert to km

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
x_v2, x_iskander, x_minuteman, x_optimus = sp.symbols('x_V2 x_Iskander x_Minuteman x_Optimus')

# Define the objective function (to be minimized) using sympy
cost_function = (rocket_data['V2']['cost_million_usd'] * x_v2 +
                 rocket_data['Iskander']['cost_million_usd'] * x_iskander +
                 rocket_data['Minuteman_III']['cost_million_usd'] * x_minuteman +
                 rocket_data[new_rocket_name]['cost_million_usd'] * x_optimus)

# Define the constraint using sympy
area_constraint = (v2_area * x_v2 + iskander_area * x_iskander + minuteman_area * x_minuteman + optimus_area * x_optimus - target_area)
# Check the type of area_constraint
print(f"\nType of area_constraint: {type(area_constraint)}")

# Perform MCMC simulation (simplified for demonstration)
num_samples = 10000
samples = []
current_state = np.array([0.25, 0.25, 0.25, 0.25])  # Initial guess (percentages of each rocket)
current_cost = cost_function.subs({x_v2: current_state[0], x_iskander: current_state[1], x_minuteman: current_state[2], x_optimus: current_state[3]})
sigma = 0.1  # Standard deviation for the proposal distribution

for i in range(num_samples):
    # Generate a proposal state
    proposal_state = current_state + np.random.normal(0, sigma, size=4)
    proposal_state = np.maximum(proposal_state, 0)  # Ensure no negative values
    proposal_state = proposal_state / np.sum(proposal_state)  # Normalize

    # Calculate the cost and constraint for the proposal state using sympy's subs
    proposal_cost = cost_function.subs({x_v2: proposal_state[0], x_iskander: proposal_state[1], x_minuteman: proposal_state[2], x_optimus: proposal_state[3]})
    proposal_area = v2_area * proposal_state[0] + iskander_area * proposal_state[1] + minuteman_area * proposal_state[2] + optimus_area * proposal_state[3]

    # Check if the proposal state satisfies the constraint (within a tolerance)
    if abs(proposal_area - target_area) < 100:  # Tolerance of 100 km^2
        # Calculate acceptance probability (simplified Metropolis-Hastings)
        acceptance_prob = min(1, np.exp(-(proposal_cost - current_cost) / 100))  # Temperature parameter = 100

        # Accept or reject the proposal
        if np.random.rand() < acceptance_prob:
            current_state = proposal_state
            current_cost = proposal_cost

    samples.append(current_state)

# Analyze the samples
samples = np.array(samples)
mean_sample = np.mean(samples, axis=0)
std_sample = np.std(samples, axis=0)

# Print the results
print("\nMCMC Simulation Results:")
print(f"Mean Proportions: V2: {mean_sample[0]:.2f}, Iskander: {mean_sample[1]:.2f}, Minuteman_III: {mean_sample[2]:.2f}, {new_rocket_name}: {mean_sample[3]:.2f}")
print(f"Std Dev Proportions: V2: {std_sample[0]:.2f}, Iskander: {std_sample[1]:.2f}, Minuteman_III: {std_sample[2]:.2f}, {new_rocket_name}: {std_sample[3]:.2f}")

# Calculate the estimated cost
estimated_cost = (rocket_data['V2']['cost_million_usd'] * mean_sample[0] +
                  rocket_data['Iskander']['cost_million_usd'] * mean_sample[1] +
                  rocket_data['Minuteman_III']['cost_million_usd'] * mean_sample[2] +
                  rocket_data[new_rocket_name]['cost_million_usd'] * mean_sample[3])
print(f"Estimated Cost to achieve {target_area} km^2: ${estimated_cost:.2f} million")
# Plotting the MCMC samples
plt.figure(figsize=(10, 6))
plt.plot(samples[:, 0], label='V2', alpha=0.5)
plt.plot(samples[:, 1], label='Iskander', alpha=0.5)
plt.plot(samples[:, 2], label='Minuteman_III', alpha=0.5)
plt.plot(samples[:, 3], label=new_rocket_name, alpha=0.5)
plt.title("MCMC Samples for Rocket Proportions")
plt.xlabel("Sample Number")
plt.ylabel("Proportion")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# =============================================================================
# 7. Plotting the Blast Radius
# =============================================================================
def plot_blast_radius(rocket_name, material_name):
    """
    Plots the blast radius of a rocket in a given material.

    Args:
        rocket_name (str): Name of the rocket.
        material_name (str): Name of the material.
    """
    # Get the rocket data
    rocket = rocket_data[rocket_name]
    energy = rocket['energy_joules']
    
    # Get the material resistance
    material_resistance = materials[material_name]
    
    # Calculate the blast radius
    radius = calculate_blast_radius(energy, material_resistance)
    
    # Plot the blast sphere
    plot_blast_sphere(radius, material_name, rocket_name)
# Example usage
plot_blast_radius('V2', 'Concrete')
plot_blast_radius('Iskander', 'Steel')
plot_blast_radius('Tomahawk', 'Concrete')
plot_blast_radius('Minuteman_III', 'Steel')
plot_blast_radius('Topol_M', 'Concrete')
plot_blast_radius('DF_41', 'Steel')
plot_blast_radius('Sarmat', 'Concrete')
plot_blast_radius(new_rocket_name, 'Steel')
# =============================================================================
# 8. Plotting the Blast Radius for All Rockets
# =============================================================================
def plot_all_blast_radii(rocket_data, material_name):
    """
    Plots the blast radius for all rockets in a given material.

    Args:
        rocket_data (dict): Dictionary containing rocket data.
        material_name (str): Name of the material.
    """
    for rocket_name, rocket in rocket_data.items():
        plot_blast_radius(rocket_name, material_name)
# Example usage
plot_all_blast_radii(rocket_data, 'Concrete')
plot_all_blast_radii(rocket_data, 'Steel')
# =============================================================================
# 9. Plotting the Blast Radius for All Rockets with Plotly
# =============================================================================
def plot_blast_radius_plotly(rocket_name, material_name):
    """
    Plots the blast radius of a rocket in a given material using Plotly.

    Args:
        rocket_name (str): Name of the rocket.
        material_name (str): Name of the material.
    """
    # Get the rocket data
    rocket = rocket_data[rocket_name]
    energy = rocket['energy_joules']
    
    # Get the material resistance
    material_resistance = materials[material_name]
    
    # Calculate the blast radius
    radius = calculate_blast_radius(energy, material_resistance)
    
    # Create a 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=10, color='red'),
        name=f"{rocket_name} in {material_name}\nBlast Radius: {radius:.2f} m"
    )])
    
    # Create a sphere for the blast radius
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        colorscale='Reds',
        opacity=0.5,
        showscale=False
    ))
    # Update layout
    fig.update_layout(
        title=f"Blast Radius of {rocket_name} in {material_name}",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.show()
# Example usage
plot_blast_radius_plotly('V2', 'Concrete')
plot_blast_radius_plotly('Iskander', 'Steel')
plot_blast_radius_plotly('Tomahawk', 'Concrete')
plot_blast_radius_plotly('Minuteman_III', 'Steel')
plot_blast_radius_plotly('Topol_M', 'Concrete')
plot_blast_radius_plotly('DF_41', 'Steel')
plot_blast_radius_plotly('Sarmat', 'Concrete')
plot_blast_radius_plotly(new_rocket_name, 'Steel')
# =============================================================================
