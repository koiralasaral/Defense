import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D # Import for 3D plotting

#  Updated rocket data with more realistic energy values (Joules)
#  Sources:
#  -   V2: Approximated from warhead yield (if available)
#  -   Iskander: Approximated from warhead type and TNT equivalent
#  -   Minuteman III: Approximated from warhead yield (W87)
#  -   Topol_M: Approximated from warhead yield
#  -   DF-41: Approximated from warhead yield
#  -   Sarmat: Approximated from warhead yield
#  -   Tomahawk: Approximated from warhead yield
rocket_data = {
    'V2': {'energy_joules': 4.184e10,  'range_km': 320, 'cost_million_usd': 0.016, 'length_m': 14, 'diameter_m': 1.65, 'cd': 0.18, 'speed_km_s': 1.4, 'type': 'Ballistic'}, # 10kT
    'Iskander': {'energy_joules': 2.5e11, 'range_km': 500, 'cost_million_usd': 2, 'length_m': 7.3, 'diameter_m': 0.92, 'cd': 0.25, 'speed_km_s': 2.1, 'type': 'Ballistic'}, # 500kg TNT
    'Minuteman_III': {'energy_joules': 3.4e14, 'range_km': 13000, 'cost_million_usd': 7, 'length_m': 18.2, 'diameter_m': 1.67, 'cd': 0.15, 'speed_km_s': 7.8, 'type': 'Ballistic'}, # 800kT
    'Topol_M': {'energy_joules': 1.0e14, 'range_km': 10000, 'cost_million_usd': 5, 'length_m': 22.7, 'diameter_m': 1.95, 'cd': 0.17, 'speed_km_s': 7.3, 'type': 'Ballistic'}, # 1MT
    'DF_41': {'energy_joules': 1.0e14, 'range_km': 14000, 'cost_million_usd': 6, 'length_m': 21, 'diameter_m': 2.0, 'cd': 0.16, 'speed_km_s': 7.0, 'type': 'Ballistic'}, # 1MT
    'Sarmat': {'energy_joules': 8.5e14, 'range_km': 18000, 'cost_million_usd': 9, 'length_m': 35.5, 'diameter_m': 3.0, 'cd': 0.19, 'speed_km_s': 7.5, 'type': 'Ballistic'}, # 20MT
    'Tomahawk': {'energy_joules': 4.184e9, 'range_km': 2500, 'cost_million_usd': 1.5, 'length_m': 5.56, 'diameter_m': 0.52, 'cd': 0.28, 'speed_km_s': 0.24, 'type': 'Cruise'} # 1kT
}

# Material properties
material_density = 11340  # kg/m^3 (Density of lead)
specific_energy = 2.4e6  # J/kg  (Specific energy for lead - approx value,  heat of fusion + some heating)

# Desired crater area
target_area_km2 = 20000  # km^2
target_area_m2 = target_area_km2 * (1000**2)  # m^2

# Function for calculating projectile motion (spherical Earth)
def projectile_motion(rocket_name, atmosphere_model='spherical'):
    """
    Calculates the trajectory of a projectile (rocket) under gravity,
    optionally considering a spherical Earth.

    Args:
        rocket_name (str): Name of the rocket (used to get parameters from rocket_data).
        atmosphere_model (str, optional): 'spherical' for spherical Earth,
                                          'flat' for flat Earth. Defaults to 'spherical'.

    Returns:
        tuple: (equation, max_altitude, inclination_angle_deg, x_traj, y_traj)
               - equation (str): Equation of the trajectory.
               - max_altitude (float): Maximum altitude reached (km).
               - inclination_angle_deg (float): Initial inclination angle (degrees).
               - x_traj (ndarray): Horizontal trajectory (km).
               - y_traj (ndarray): Vertical trajectory (km).
    """
    # Constants
    g = 9.80665  # m/s^2
    R = 6371  # km (radius of Earth)

    # Rocket parameters
    if rocket_name in rocket_data:
        v0 = rocket_data[rocket_name]['speed_km_s'] * 1000  # m/s
    elif rocket_name == 'Chimera':
        v0 = new_rocket_data['Chimera']['speed_km_s'] * 1000
    else:
        print(f"Warning: Rocket data for {rocket_name} not found. Using default speed.")
        v0 = 0
    inclination_angle_deg = 45  # degrees
    inclination_angle_rad = np.radians(inclination_angle_deg)

    # Initial conditions
    x0 = 0
    y0 = 0

    if atmosphere_model == 'spherical':
        # Spherical Earth model
        def get_trajectory(theta):
            """Calculate trajectory for a given angle."""
            # Use the approximation for range on a spherical Earth
            range_km = (2 * R * v0**2 * np.sin(2*theta)) / (g * np.cos(theta) + 2 * v0**2 * np.sin(theta)/R )/1000
            #Max altitude h = (v0^2 * sin^2(theta))/(2*g)
            max_altitude_km = (v0**2 * np.sin(theta)**2) / (2 * g) / 1000
            # Number of points
            num_points = 1000
            # Generate x and y values
            x_traj = np.linspace(0, range_km, num_points)
            y_traj = []

            for x in x_traj:
                # Iteratively solve for y, accounting for the Earth's curvature
                y_guess = max_altitude_km  # Initial guess
                for _ in range(10):  # Iterate to improve the guess, avoid infinite loop.
                  y_new = (v0 * np.sin(theta) / (v0 * np.cos(theta)) )* x - (g * x**2) / (2 * v0**2 * np.cos(theta)**2)
                  y_guess = y_new
                y_traj.append(y_guess)
            return x_traj, np.array(y_traj), max_altitude_km, range_km

        x_traj, y_traj, max_altitude_km, range_km = get_trajectory(inclination_angle_rad)
        equation = "Spherical Earth Trajectory" # Placeholder, the exact equation is complex
        return equation, max_altitude_km, inclination_angle_deg, x_traj, y_traj

    elif atmosphere_model == 'flat':
        # Flat Earth model (for comparison)
        t_flight = (2 * v0 * np.sin(inclination_angle_rad)) / g
        max_altitude_km = (v0**2 * np.sin(inclination_angle_rad)**2) / (2 * g) / 1000
        x_traj = np.linspace(0, (v0 * np.cos(inclination_angle_rad) * t_flight) / 1000, 1000)
        y_traj = (v0 * np.sin(inclination_angle_rad) * x_traj * 1000 / (v0 * np.cos(inclination_angle_rad))) - (0.5 * g * (x_traj * 1000 / (v0 * np.cos(inclination_angle_rad)))**2) / 1000
        equation = f"y = {v0*np.sin(inclination_angle_rad):.2f}x - {0.5*g:.2f}x^2"
        return equation, max_altitude_km, inclination_angle_deg, x_traj, y_traj
    else:
        raise ValueError("Invalid atmosphere_model. Choose 'spherical' or 'flat'.")



# =============================================================================
# 8. Linear Regression for Blast Damage Area
# =============================================================================

print("\nLinear Regression for Blast Damage Area:")

# Prepare the data for linear regression
X = np.array([rocket_data['V2']['energy_joules'],
                 rocket_data['Iskander']['energy_joules'],
                 rocket_data['Minuteman_III']['energy_joules'],
                 rocket_data['Tomahawk']['energy_joules']]).reshape(-1, 1)  # Independent variable: Energy
y = np.array([500, 1200, 10000, 15000]) # Dependent variable: Blast Area

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, y)

# Get the coefficients
slope = model.coef_[0]
intercept = model.intercept_

print(f"Linear Regression Equation: Blast Area = {slope:.2e} * Energy + {intercept:.2f}")

# Predict the energy required for the target area
predicted_energy = (target_area_m2 - intercept) / slope
print(f"Predicted Energy to achieve {target_area_km2} km^2: {predicted_energy:.2e} Joules")

# Calculate R-squared
r_squared = model.score(X, y)
print(f"R-squared: {r_squared:.2f}")

# Plot the data and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, label="Rocket Data")
plt.plot(X, model.predict(X), color='red', label=f"Regression Line (R^2 = {r_squared:.2f})")
plt.xlabel("Energy (Joules)")
plt.ylabel("Blast Damage Area (km^2)")
plt.title("Linear Regression of Blast Area vs. Energy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# 9. Designing a New Missile using Linear Regression
# =============================================================================
print("\nDesigning a New Missile using Linear Regression:")

# Use the regression equation to estimate the energy for a desired blast area
desired_blast_area = 25000  # km^2 (Example desired blast area)
estimated_energy = (desired_blast_area - intercept) / slope

#  Now, let's estimate other properties based on the existing rockets.  We'll use a simplified approach.
#  A more sophisticated approach might involve more complex modeling or constraints.

# Average properties of existing rockets (excluding Tomahawk as it is a cruise missile)
valid_rockets = ['V2', 'Iskander', 'Minuteman_III', 'Topol_M', 'DF_41', 'Sarmat']
avg_range = np.mean([rocket_data[r]['range_km'] for r in valid_rockets])
avg_cost = np.mean([rocket_data[r]['cost_million_usd'] for r in valid_rockets])
avg_length = np.mean([rocket_data[r]['length_m'] for r in valid_rockets])
avg_diameter = np.mean([rocket_data[r]['diameter_m'] for r in valid_rockets])
avg_cd = np.mean([rocket_data[r]['cd'] for r in valid_rockets])
avg_speed = np.mean([rocket_data[r]['speed_km_s'] for r in valid_rockets])

# New missile design - A simple approach is to scale from the averages, with a bit of randomness
new_missile_name = "Chimera"
new_rocket_data = {}
new_rocket_data[new_missile_name] = {
    "range_km": avg_range * (1.1 + np.random.uniform(-0.1, 0.1)),  # Slightly vary range
    "tnt_equivalent_kg": estimated_energy / 4.184e6,
    "cd": avg_cd * (1.0 + np.random.uniform(-0.1, 0.1)),
    "speed_km_s": avg_speed * (1.0 + np.random.uniform(-0.1, 0.1)),
    "length_m": avg_length * (1.0 + np.random.uniform(-0.1, 0.1)),
    "diameter_m": avg_diameter * (1.0 + np.random.uniform(-0.1, 0.1)),
    "type": "Composite",  # Could be a new type
    "cost_million_usd": avg_cost * (1.2 + np.random.uniform(-0.2, 0.2)), # Higher cost, since its better
    "energy_joules": estimated_energy,
}

print(f"\nNew Missile Design: {new_missile_name}")
for prop, value in new_rocket_data[new_missile_name].items():
    if isinstance(value, (int, float)):
        print(f"  {prop}: {value:.2f}")
    else:
        print(f"  {prop}: {value}") # print the value without formatting

# Plot the trajectory of the new missile
equation, max_altitude, inclination_angle_deg, x_traj, y_traj = projectile_motion(new_missile_name, atmosphere_model='spherical')
print(f"\n{new_missile_name} Trajectory Equation: {equation}")
print(f"  Max Altitude: {max_altitude:.2f} km")
print(f"  Initial Inclination Angle: {inclination_angle_deg:.2f} deg")
# Calculate blast area
blast_area = slope * new_rocket_data[new_missile_name]['energy_joules'] + intercept

plt.figure(figsize=(8, 6))
plt.plot(x_traj, y_traj, label=f"{new_missile_name} Trajectory (Spherical Earth)")
plt.xlabel("Horizontal Distance (km)")
plt.ylabel("Altitude (km)")
plt.title(f"{new_missile_name} Trajectory\nBlast Area: {blast_area:.2f} km^2") # Added blast area to title
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# 10. Supremum Norm and Refined Missile Specs
# =============================================================================
print("\nSupremum Norm and Refined Missile Specs:")

# 1. Calculate Trajectory Points for Existing Rockets
trajectory_points = {}
for rocket_name in valid_rockets:
    _, _, _, x_coords, y_coords = projectile_motion(rocket_name, atmosphere_model='spherical')
    trajectory_points[rocket_name] = np.column_stack((x_coords, y_coords))

# 2. Calculate Trajectory Points for New Missile
_, _, _, new_x_coords, new_y_coords = projectile_motion(new_missile_name, atmosphere_model='spherical')
new_trajectory_points = np.column_stack((new_x_coords, new_y_coords))

# 3. Resample Trajectories to Have the Same Number of Points
num_points = 1000  # Increased for better comparison
resampled_trajectories = {}
for rocket_name in valid_rockets:
    x_interp = np.interp(np.linspace(0, 1, num_points), trajectory_points[rocket_name][:, 0], np.linspace(0, 1, trajectory_points[rocket_name].shape[0]))
    y_interp = np.interp(np.linspace(0, 1, num_points), trajectory_points[rocket_name][:, 1], np.linspace(0, 1, trajectory_points[rocket_name].shape[0]))
    resampled_trajectories[rocket_name] = np.column_stack((x_interp, y_interp))

new_x_interp = np.interp(np.linspace(0, 1, num_points), new_trajectory_points[:, 0], np.linspace(0, 1, new_trajectory_points.shape[0]))
new_y_interp = np.interp(np.linspace(0, 1, num_points), new_trajectory_points[:, 1], np.linspace(0, 1, new_trajectory_points.shape[0]))
resampled_new_trajectory = np.column_stack((new_x_interp, new_y_interp))

# 4. Calculate Supremum Norm
supremum_norms = []
for rocket_name in valid_rockets:
    diff = resampled_new_trajectory - resampled_trajectories[rocket_name]
    norm = np.max(np.sqrt(np.sum(diff**2, axis=1)))  # Euclidean norm for each point, then max
    supremum_norms.append(norm)

# 5. Find the Rocket with the Minimum Supremum Norm
min_norm_index = np.argmin(supremum_norms)
closest_rocket = valid_rockets[min_norm_index]
min_supremum_norm = supremum_norms[min_norm_index]

print(f"The closest rocket in trajectory to {new_missile_name} is {closest_rocket} with a supremum norm of {min_supremum_norm:.2f} km.")

# 6. Refine New Missile Specs based on Closest Rocket
# This is a simplified approach.  A more advanced method might involve interpolation or weighted averages.
closest_rocket_data = rocket_data[closest_rocket]

# Define a scaling factor
scale_factor = 1.05  # Adjust as needed

# Refine the new missile specs, ensuring no zero values
new_rocket_data[new_missile_name]["range_km"] = max(1, closest_rocket_data["range_km"] * scale_factor)
new_rocket_data[new_missile_name]["tnt_equivalent_kg"] = max(1, estimated_energy / 4.184e6)
new_rocket_data[new_missile_name]["cd"] = max(0.01, closest_rocket_data["cd"] * scale_factor)
new_rocket_data[new_missile_name]["speed_km_s"] = max(1, closest_rocket_data["speed_km_s"] * scale_factor)
new_rocket_data[new_missile_name]["length_m"] = max(1, closest_rocket_data["length_m"] * scale_factor)
new_rocket_data[new_missile_name]["diameter_m"] = max(1, closest_rocket_data["diameter_m"] * scale_factor)
new_rocket_data[new_missile_name]["cost_million_usd"] = max(1, closest_rocket_data["cost_million_usd"] * scale_factor)
new_rocket_data[new_missile_name]["energy_joules"] = max(1, estimated_energy)  # Ensure energy is not zero

# Calculate blast area and add it to the dictionary
blast_area = slope * new_rocket_data[new_missile_name]['energy_joules'] + intercept
new_rocket_data[new_missile_name]["blast_area_km2"] = blast_area

print(f"\nRefined {new_missile_name} Specs based on {closest_rocket}:")
for prop, value in new_rocket_data[new_missile_name].items():
    if isinstance(value, (int, float)):
        print(f"  {prop}: {value:.2f}")
    else:
        print(f"  {prop}: {value}") # print the value without formatting
# 7.  3D plot of the Blast Area
def plot_3d_blast_area(energy_joules, title="3D Blast Area"):
    """
    Plots the 3D blast area as a sphere, with the radius determined by the energy.

    Args:
        energy_joules (float): Energy of the explosion in Joules.
        title (str): Title of the plot
    """
    material_strength = 1e7  # Example material strength
    radius_m = (energy_joules / material_strength)**(1/3)
    radius_km = radius_m / 1000

    # Create the sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius_km * np.cos(u)*np.sin(v)
    y = radius_km * np.sin(u)*np.sin(v)
    z = radius_km * np.cos(v)

    # Create the figure and axes
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z, color='red', alpha=0.5)
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# Plot the 3D blast area for the new missile
plot_3d_blast_area(new_rocket_data[new_missile_name]['energy_joules'], title=f"3D Blast Area for {new_missile_name}")

# =============================================================================
# 11. Rockets Required for Target Crater
# =============================================================================
print("\nRockets Required for Target Crater:")

# Calculate the radius of the desired crater, assuming it's circular
target_radius_m = np.sqrt(target_area_m2 / np.pi)

# Calculate the volume of the desired crater (hemisphere approximation)
crater_volume = (2/3) * np.pi * (target_radius_m ** 3)  # m^3

# Calculate the mass of lead in the crater
crater_mass = material_density * crater_volume  # kg

# Calculate the total energy required to create the crater
total_energy_required = crater_mass * specific_energy  # J

print(f"Total energy required to create a crater of 20000 km^2 area in lead: {total_energy_required:.2e} Joules")

# Calculate the number of rockets of each type required
rockets_required = {}
for rocket_name, data in rocket_data.items():
    rockets_required[rocket_name] = np.ceil(total_energy_required / data['energy_joules'])  # Use ceil to ensure enough energy
    print(f"{rockets_required[rocket_name]:.0f} {rocket_name} rockets are required.")
