import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math # For math.radians
import pydeck as pdk # Import pydeck
import pandas as pd # Import pandas for DataFrame
import matplotlib.colors # For converting named named colors to RGBA
from scipy.integrate import solve_ivp # Required for projectile motion
from scipy.stats import multivariate_normal # For MCMC target distribution
import folium # Import folium

# =============================================================================
# 0. Global Configuration & Constants (Copied from rocket_simulation_script)
# =============================================================================
# Target ranges for specific rockets (km)
TARGET_RANGES_KM = {
    "V2": 320,
    "Iskander": 500,
    "Tomahawk": 2500,
    "Minuteman_III": 13000,
    "Topol_M": 11000,
    "DF_41": 14000,
    "Sarmat": 18000,
}

# Material strengths for crater calculation (Pa)
MATERIAL_STRENGTHS = {
    "Lead": 2e7,
    "Steel": 2.5e8,
    "Rock": 1.5e8,
    "Earth": 1e6,
    "Concrete": 3e7
}
DEFAULT_MATERIAL_STRENGTH_FOR_SORTING = MATERIAL_STRENGTHS["Earth"]

# Animation parameters
NUM_ANIMATION_FRAMES = 300
ANIMATION_INTERVAL_MS = 30

# Scaling for velocity and acceleration arrows in Matplotlib
ARROW_LENGTH_SCALE = 50000 # meters, roughly 50 km for visibility
FLUX_MARKER_SCALE = 100 # meters, for PyDeck scatterplot radius based on flux magnitude

# =============================================================================
# 1. Data Collection and Preparation (Copied from rocket_simulation_script)
# =============================================================================

rocket_data = {
    "V2": {
        "range_km": 320,
        "tnt_equivalent_kg": 1000,
        "cd": 0.15,
        "speed_km_s": 1.6,
        "length_m": 14,
        "diameter_m": 1.65,
        "type": "Ballistic",
        "cost_million_usd": 0.3,
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
        "tnt_equivalent_kg": 300000,
        "cd": 0.20,
        "speed_km_s": 7,
        "length_m": 18.2,
        "diameter_m": 1.67,
        "type": "ICBM",
        "cost_million_usd": 7
    },
    "Topol_M": {
        "range_km": 11000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.22,
        "speed_km_s": 7.3,
        "length_m": 22.7,
        "diameter_m": 1.86,
        "type": "ICBM",
        "cost_million_usd": 8
    },
    "DF_41": {
        "range_km": 14000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.21,
        "speed_km_s": 7.8,
        "length_m": 16.5,
        "diameter_m": 2,
        "type": "ICBM",
        "cost_million_usd": 10
    },
    "Sarmat":{
        "range_km": 18000,
        "tnt_equivalent_kg": 8000000,
        "cd": 0.23,
        "speed_km_s": 7.5,
        "length_m": 35.5,
        "diameter_m": 3,
        "type": "ICBM",
        "cost_million_usd": 20
    }
}

# Convert TNT equivalent to Joules
for rocket_name in rocket_data:
    rocket_data[rocket_name]["energy_joules"] = rocket_data[rocket_name]["tnt_equivalent_kg"] * 4.184e6

# =============================================================================
# 2. Projectile Motion Analysis (Copied from rocket_simulation_script)
# =============================================================================
def projectile_motion(rocket_name_pm, current_speed_m_s, launch_angle_deg=45, atmosphere_model='spherical'):
    """
    Calculates the projectile motion of a rocket.
    Uses current_speed_m_s passed as argument instead of from rocket_data directly for adjustment purposes.
    Returns x, y, vx, vy, ax, ay at each time step.
    """
    rocket = rocket_data[rocket_name_pm]

    # Constants
    g_0 = 9.81
    rho_0 = 1.225
    R_earth = 6371e3
    M_earth = 5.972e24
    G = 6.674e-11
    H_scale = 8500

    A = np.pi * (rocket["diameter_m"] / 2) ** 2
    Cd = rocket["cd"]
    m = rocket["tnt_equivalent_kg"]
    if m <= 0:
        m = 1000

    v0 = current_speed_m_s
    initial_angle_rad = np.radians(launch_angle_deg)

    def ode_system(t, state):
        x, y, vx, vy = state
        v_mag = np.sqrt(vx**2 + vy**2)

        if atmosphere_model == 'spherical':
            current_altitude = y
            r_grav = R_earth + y
            g = G * M_earth / r_grav**2 if r_grav > 0 else g_0
            rho_at_altitude = rho_0 * np.exp(-current_altitude / H_scale)
            rho_at_altitude = max(0, rho_at_altitude)
        else:
            g = g_0
            rho_at_altitude = rho_0

        drag = 0.5 * rho_at_altitude * Cd * A * v_mag**2

        if v_mag != 0 and m > 0:
            ax = -drag * vx / (m * v_mag)
            ay = -g - (drag * vy / (m * v_mag))
        elif m > 0:
            ax = 0
            ay = -g
        else:
            ax = 0
            ay = 0

        return [vx, vy, ax, ay]

    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    y0_state = [0, 0, v0 * np.cos(initial_angle_rad), v0 * np.sin(initial_angle_rad)]

    t_eval_max = 7000
    max_step_val = 1.0

    sol = solve_ivp(ode_system, (0, t_eval_max), y0_state, events=hit_ground, dense_output=True, max_step=max_step_val)

    t_vals = sol.t
    x_vals = sol.y[0]
    y_vals = sol.y[1]
    vx_vals = sol.y[2]
    vy_vals = sol.y[3]

    # Calculate acceleration components
    ax_vals = np.zeros_like(t_vals)
    ay_vals = np.zeros_like(t_vals)

    for i in range(len(t_vals)):
        current_state = [x_vals[i], y_vals[i], vx_vals[i], vy_vals[i]]
        # Call ode_system to get accelerations at each point
        _, _, current_ax, current_ay = ode_system(t_vals[i], current_state)
        ax_vals[i] = current_ax
        ay_vals[i] = current_ay

    mask = y_vals >= -1
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    vx_vals = vx_vals[mask]
    vy_vals = vy_vals[mask]
    ax_vals = ax_vals[mask]
    ay_vals = ay_vals[mask]

    if len(y_vals) > 0:
        y_vals[-1] = max(y_vals[-1], 0)

    equation_str = "N/A"
    if len(x_vals) > 2:
        try:
            coeffs = np.polyfit(x_vals, y_vals, 2)
            equation_str = f"y = {coeffs[0]:.4e}x^2 + {coeffs[1]:.4e}x + {coeffs[2]:.4e}"
        except (np.linalg.LinAlgError, ValueError):
            equation_str = "Fit failed."

    max_altitude = np.max(y_vals) / 1000 if len(y_vals) > 0 else 0
    achieved_range_m = np.max(x_vals) if len(x_vals) > 0 else 0

    return x_vals, y_vals, equation_str, max_altitude, launch_angle_deg, achieved_range_m, vx_vals, vy_vals, ax_vals, ay_vals

def adjust_speed_for_range(rocket_name_adj, target_range_km, initial_launch_angle_deg=45):
    """
    Adjusts rocket speed to achieve the target range.
    Updates rocket_data with the new speed.
    """
    rocket = rocket_data[rocket_name_adj]
    current_speed_km_s = rocket['speed_km_s']
    
    tolerance_km = target_range_km * 0.01
    max_iterations = 30
    
    best_achieved_range_km = -1
    best_speed_km_s = current_speed_km_s
    best_x_vals, best_y_vals = np.array([]), np.array([])
    best_vx_vals, best_vy_vals, best_ax_vals, best_ay_vals = np.array([]), np.array([]), np.array([]), np.array([])

    low_speed_km_s = 0.1
    high_speed_km_s = 30

    for iteration in range(max_iterations):
        current_speed_m_s = current_speed_km_s * 1000
        
        x_vals, y_vals, _, _, _, achieved_range_m, vx_vals, vy_vals, ax_vals, ay_vals = projectile_motion(
            rocket_name_adj, current_speed_m_s, initial_launch_angle_deg, atmosphere_model='spherical'
        )
        achieved_range_km = achieved_range_m / 1000

        if best_achieved_range_km == -1 or abs(achieved_range_km - target_range_km) < abs(best_achieved_range_km - target_range_km):
            best_achieved_range_km = achieved_range_km
            best_speed_km_s = current_speed_km_s
            best_x_vals, best_y_vals = x_vals, y_vals
            best_vx_vals, best_vy_vals, best_ax_vals, best_ay_vals = vx_vals, vy_vals, ax_vals, ay_vals

        error_km = achieved_range_km - target_range_km

        if abs(error_km) < tolerance_km:
            rocket['speed_km_s'] = current_speed_km_s
            return x_vals, y_vals, current_speed_km_s, achieved_range_km, vx_vals, vy_vals, ax_vals, ay_vals

        if achieved_range_km > 0:
            scale_factor = (target_range_km / achieved_range_km)**0.5 
            scale_factor = np.clip(scale_factor, 0.85, 1.15)
            current_speed_km_s *= scale_factor
        else:
            current_speed_km_s *= 1.5

        current_speed_km_s = np.clip(current_speed_km_s, low_speed_km_s, high_speed_km_s)
        
        if iteration > 5 and current_speed_km_s == low_speed_km_s and error_km < 0:
             break
        if iteration > 5 and current_speed_km_s == high_speed_km_s and error_km > 0:
             break

    rocket['speed_km_s'] = best_speed_km_s
    return best_x_vals, best_y_vals, best_speed_km_s, best_achieved_range_km, best_vx_vals, best_vy_vals, best_ax_vals, best_ay_vals


# =============================================================================
# 3. Atmospheric Layer and Nepal Map Definitions
# =============================================================================

# Nepal Latitude and Longitude Bounds (approximate)
NEPAL_LAT_MIN = 26.3
NEPAL_LAT_MAX = 30.5
NEPAL_LON_MIN = 80.0
NEPAL_LON_MAX = 88.2

# New global constants for Nepal center
NEPAL_LAT_CENTER = (NEPAL_LAT_MIN + NEPAL_LAT_MAX) / 2
NEPAL_LON_CENTER = (NEPAL_LON_MIN + NEPAL_LON_MAX) / 2

# Global Earth Radius
R_EARTH = 6371e3 # meters

# Conversion factors (approximate)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_AT_NEPAL_AVG_LAT = 111.0 * math.cos(math.radians(NEPAL_LAT_CENTER))

# Convert lat/lon bounds to meters for plotting (local Cartesian for MCMC)
NEPAL_X_MIN = 0
NEPAL_X_MAX = (NEPAL_LON_MAX - NEPAL_LON_MIN) * KM_PER_DEG_LON_AT_NEPAL_AVG_LAT * 1000
NEPAL_Y_MIN = 0
NEPAL_Y_MAX = (NEPAL_LAT_MAX - NEPAL_LAT_MIN) * KM_PER_DEG_LAT * 1000

# Atmospheric Layer Altitudes (in meters)
ATMOSPHERE_LAYERS = {
    "Mini-Troposphere": {"min_alt": 0, "max_alt": 2000, "color": 'lightblue', "alpha": 0.2},
    "Troposphere": {"min_alt": 2000, "max_alt": 12000, "color": 'skyblue', "alpha": 0.2},
    "Stratosphere": {"min_alt": 12000, "max_alt": 50000, "color": 'lightgreen', "alpha": 0.15},
    "Thermosphere": {"min_alt": 50000, "max_alt": 500000, "color": 'lightcoral', "alpha": 0.1},
    "Exosphere": {"min_alt": 500000, "max_alt": 1000000, "color": 'lavender', "alpha": 0.05} # Visual upper limit
}

# Hardcoded simplified Nepal boundary coordinates (Lon, Lat)
# This is a very simplified polygon to represent Nepal's shape
NEPAL_BOUNDARY_COORDS_LON_LAT = [
    [80.0, 26.5], [80.5, 27.0], [81.5, 27.5], [82.5, 28.0], [83.5, 28.5],
    [84.5, 28.8], [85.5, 28.9], [86.5, 28.8], [87.5, 28.5], [88.0, 28.0],
    [88.0, 27.5], [87.5, 27.0], [87.0, 26.5], [86.0, 26.3], [85.0, 26.3],
    [84.0, 26.5], [83.0, 26.5], [82.0, 26.3], [81.0, 26.3], [80.0, 26.5]
]


def convert_lat_lon_alt_to_cartesian(lat_deg, lon_deg, alt_m):
    """Converts spherical coordinates (degrees) to Cartesian (meters) relative to Earth's center."""
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    r = R_EARTH + alt_m

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z

def convert_local_meters_to_global_lat_lon_alt(local_x_meters, local_y_meters, altitude_meters, launch_lat_deg, launch_lon_deg):
    """
    Converts local Cartesian coordinates (meters from a launch point on Earth's surface)
    to global latitude, longitude, and altitude.
    Assumes local_x_meters is roughly East-West, local_y_meters is roughly North-South.
    """
    # Approximate conversion of meters to degrees lat/lon offset
    delta_lat_deg = local_y_meters / (KM_PER_DEG_LAT * 1000)
    # Longitude conversion depends on latitude, use launch_lat for approximation
    delta_lon_deg = local_x_meters / (KM_PER_DEG_LON_AT_NEPAL_AVG_LAT * 1000) # Use average Nepal lat for simplicity

    global_lat_deg = launch_lat_deg + delta_lat_deg
    global_lon_deg = launch_lon_deg + delta_lon_deg
    
    return global_lat_deg, global_lon_deg, altitude_meters

def convert_global_lat_lon_to_local_meters(lat_deg, lon_deg, origin_lat_deg, origin_lon_deg):
    """
    Converts global latitude/longitude to local meters relative to an origin point.
    Used for mapping real Nepal boundary onto the 2D MCMC plot.
    """
    delta_lat_deg = lat_deg - origin_lat_deg
    delta_lon_deg = lon_deg - origin_lon_deg

    local_y_meters = delta_lat_deg * KM_PER_DEG_LAT * 1000
    local_x_meters = delta_lon_deg * KM_PER_DEG_LON_AT_NEPAL_AVG_LAT * 1000 # Use average Nepal lat for conversion

    return local_x_meters, local_y_meters

def find_optimal_angle_for_max_range_at_given_speed(rocket_name, speed_m_s, min_angle=20, max_angle=70, angle_step=1):
    """
    Finds the optimal launch angle that maximizes range for a given rocket at a specific speed.
    Returns the optimal angle (deg) and the maximum range (m) achieved.
    """
    max_range_m = 0
    optimal_angle_deg = 45 # Default

    angles_to_test = np.arange(min_angle, max_angle + angle_step, angle_step)

    for angle in angles_to_test:
        x_vals, y_vals, _, _, _, achieved_range_m, _, _, _, _ = projectile_motion( # Only need achieved_range_m
            rocket_name, speed_m_s, angle, atmosphere_model='spherical'
        )
        if achieved_range_m > max_range_m:
            max_range_m = achieved_range_m
            optimal_angle_deg = angle
    return optimal_angle_deg, max_range_m

def calculate_crater_dimensions(energy_joules, material_strength_pa):
    """
    Calculates approximate crater radius and depth based on explosion energy and material strength.
    Simplified model based on scaling laws.
    """
    # Constants for cratering (empirical, can vary widely)
    k_crater = 0.001 # Scaling constant
    exponent = 1/3.4 # Typical exponent for energy scaling

    # Convert energy from Joules to TNT equivalent (already done in rocket_data)
    # E_TNT_kg = energy_joules / 4.184e6

    # Convert material strength from Pa to MPa for common formulas (1 MPa = 1e6 Pa)
    material_strength_MPa = material_strength_pa / 1e6

    # Simplified formula for crater radius (meters)
    # This is a very rough approximation. More complex models exist.
    crater_radius_m = k_crater * (energy_joules / material_strength_MPa)**exponent

    # Crater depth is often a fraction of the radius, e.g., 1/3 to 1/5
    crater_depth_m = crater_radius_m / 4.0 # Roughly 1/4 of radius

    return crater_radius_m, crater_depth_m


# =============================================================================
# 4. Main Animation Logic (Matplotlib)
# =============================================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set initial camera view
ax.view_init(elev=20, azim=20)

# Earth Sphere
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 50) # Full sphere for Earth
X_earth = R_EARTH * np.outer(np.cos(u), np.sin(v))
Y_earth = R_EARTH * np.outer(np.sin(u), np.sin(v))
Z_earth = R_EARTH * np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(X_earth, Y_earth, Z_earth, color='lightgray', alpha=0.5, label='Earth Sphere', rstride=4, cstride=4)

# Plot Nepal region on Earth's surface using hardcoded boundary
nepal_surface_x, nepal_surface_y, nepal_surface_z = [], [], []
for lon, lat in NEPAL_BOUNDARY_COORDS_LON_LAT:
    x, y, z = convert_lat_lon_alt_to_cartesian(lat, lon, 0)
    nepal_surface_x.append(x)
    nepal_surface_y.append(y)
    nepal_surface_z.append(z)
ax.plot(nepal_surface_x, nepal_surface_y, nepal_surface_z, color='darkgreen', linewidth=2, label='Nepal Region')


# Satellite position (animated to orbit slightly)
satellite_alt_m = 400e3 # 400 km
# Initial satellite position (fixed for now, will animate later)
sat_lat_init, sat_lon_init = NEPAL_LAT_CENTER + 2, NEPAL_LON_CENTER + 2 # Slightly offset from Nepal center
sat_x_init, sat_y_init, sat_z_init = convert_lat_lon_alt_to_cartesian(sat_lat_init, sat_lon_init, satellite_alt_m)
satellite_plot, = ax.plot([sat_x_init], [sat_y_init], [sat_z_init], 's', color='cyan', markersize=8, label='Satellite') # 's' for square marker


# Plot Atmospheric Layers (as spherical segments over Nepal)
lat_grid_layers = np.linspace(NEPAL_LAT_MIN, NEPAL_LAT_MAX, 20)
lon_grid_layers = np.linspace(NEPAL_LON_MIN, NEPAL_LON_MAX, 20)
LON_MESH_LAYERS, LAT_MESH_LAYERS = np.meshgrid(lon_grid_layers, lat_grid_layers)

for layer_name, params in ATMOSPHERE_LAYERS.items():
    min_alt = params["min_alt"]
    max_alt = params["max_alt"]
    color = params["color"]
    alpha = params["alpha"]

    # Bottom surface of the layer (spherical segment)
    X_bottom, Y_bottom, Z_bottom = convert_lat_lon_alt_to_cartesian(LAT_MESH_LAYERS, LON_MESH_LAYERS, min_alt)
    ax.plot_surface(X_bottom, Y_bottom, Z_bottom, color=color, alpha=alpha, shade=False, label=f'{layer_name} Bottom')
    
    # Top surface of the layer (spherical segment)
    X_top, Y_top, Z_top = convert_lat_lon_alt_to_cartesian(LAT_MESH_LAYERS, LON_MESH_LAYERS, max_alt)
    ax.plot_surface(X_top, Y_top, Z_top, color=color, alpha=alpha, shade=False, label=f'{layer_name} Top')


# Define missiles and their target ranges (already adjusted in previous script)
# We will use the speeds already adjusted by the previous script for these rockets.
missiles_to_simulate = {
    "V2": {"color": 'red', "launch_angle": 60, "launch_azimuth": 45}, # Added azimuth
    "Iskander": {"color": 'orange', "launch_angle": 55, "launch_azimuth": 135},
    "Minuteman_III": {"color": 'blue', "launch_angle": 45, "launch_azimuth": 225},
    "DF_41": {"color": 'purple', "launch_angle": 40, "launch_azimuth": 315},
    "Sarmat": {"color": 'black', "launch_angle": 35, "launch_azimuth": 0}, # Launching North
}

trajectory_data = {}
trajectory_lines = {}
trajectory_points = {}
collision_markers = {}
crater_plots = {}
velocity_arrows = {}
acceleration_arrows = {}
normal_vectors_at_entry = {} # To store normal vectors for plotting
flux_markers = {} # To store markers for flux visualization

# Pre-calculate trajectories and collision points
for rocket_name, params in missiles_to_simulate.items():
    if rocket_name not in rocket_data:
        print(f"Warning: {rocket_name} not found in rocket_data. Skipping trajectory calculation.")
        continue

    # Find optimal angle and adjust speed for maximum range
    optimal_angle_for_max_range, max_achievable_range_m = find_optimal_angle_for_max_range_at_given_speed(
        rocket_name, rocket_data[rocket_name]['speed_km_s'] * 1000, # Convert km/s to m/s
        min_angle=20, max_angle=70, angle_step=1
    )

    x_traj_2d, y_traj_2d, final_speed, achieved_range, vx_vals_2d, vy_vals_2d, ax_vals_2d, ay_vals_2d = adjust_speed_for_range(
        rocket_name, max_achievable_range_m / 1000, # Convert meters to km for target range
        initial_launch_angle_deg=optimal_angle_for_max_range
    )
    
    # For visualization, we will launch from the center of Nepal for global projection
    launch_lat_deg = NEPAL_LAT_CENTER
    launch_lon_deg = NEPAL_LON_CENTER
    launch_azimuth_rad = np.radians(params["launch_azimuth"])

    # Project 2D trajectory (range, altitude) onto 3D Earth with azimuth
    global_lats, global_lons, global_alts = [], [], []
    global_X, global_Y, global_Z = [], [], []
    
    # Store velocity and acceleration vectors in global Cartesian coordinates
    global_vx, global_vy, global_vz = [], [], []
    global_ax, global_ay, global_az = [], [], []

    for i in range(len(x_traj_2d)):
        # Calculate local horizontal displacement based on range and azimuth
        local_x_disp = x_traj_2d[i] * np.sin(launch_azimuth_rad) # East-West component
        local_y_disp = x_traj_2d[i] * np.cos(launch_azimuth_rad) # North-South component
        
        lat, lon, alt = convert_local_meters_to_global_lat_lon_alt(local_x_disp, local_y_disp, y_traj_2d[i], launch_lat_deg, launch_lon_deg)
        global_lats.append(lat)
        global_lons.append(lon)
        global_alts.append(alt)

        x, y, z = convert_lat_lon_alt_to_cartesian(lat, lon, alt)
        global_X.append(x)
        global_Y.append(y)
        global_Z.append(z)

        # Convert 2D velocity (vx_2d, vy_2d) to 3D global Cartesian velocity
        # vx_2d is tangential speed along range, vy_2d is vertical speed
        # We need to project vx_2d onto the local East-West and North-South axes
        v_horizontal_magnitude = vx_vals_2d[i]
        gv_x = v_horizontal_magnitude * np.sin(launch_azimuth_rad) # East-West component
        gv_y = v_horizontal_magnitude * np.cos(launch_azimuth_rad) # North-South component
        gv_z = vy_vals_2d[i] # Vertical component (altitude change)
        global_vx.append(gv_x)
        global_vy.append(gv_y)
        global_vz.append(gv_z)

        # Convert 2D acceleration (ax_2d, ay_2d) to 3D global Cartesian acceleration
        a_horizontal_magnitude = ax_vals_2d[i]
        ga_x = a_horizontal_magnitude * np.sin(launch_azimuth_rad)
        ga_y = a_horizontal_magnitude * np.cos(launch_azimuth_rad)
        ga_z = ay_vals_2d[i]
        global_ax.append(ga_x)
        global_ay.append(ga_y)
        global_az.append(ga_z)


    # Calculate local coordinates for MCMC based on the final impact point
    final_impact_global_lat = global_lats[-1]
    final_impact_global_lon = global_lons[-1]
    
    # Assuming MCMC target is relative to NEPAL_LAT_MIN, NEPAL_LON_MIN
    final_impact_local_x_mcmc, final_impact_local_y_mcmc = convert_global_lat_lon_to_local_meters(
        final_impact_global_lat, final_impact_global_lon, NEPAL_LAT_MIN, NEPAL_LON_MIN
    )


    trajectory_data[rocket_name] = {
        'x_global': np.array(global_X),
        'y_global': np.array(global_Y),
        'z_global': np.array(global_Z),
        'global_lats': np.array(global_lats), # Store for Folium
        'global_lons': np.array(global_lons), # Store for Folium
        'global_alts': np.array(global_alts), # Store for Folium
        'x_local_mcmc': final_impact_local_x_mcmc, # Local x for MCMC target
        'y_local_mcmc': final_impact_local_y_mcmc, # Local y for MCMC target
        'z_local_altitude': y_traj_2d, # Altitude for local context
        'color': params["color"],
        'length': len(x_traj_2d),
        'final_impact_x_local': final_impact_local_x_mcmc, # Use local for MCMC target
        'final_impact_y_local': final_impact_local_y_mcmc, # Use local for MCMC target
        'final_impact_global_lat': global_lats[-1],
        'final_impact_global_lon': global_lons[-1],
        'energy_joules': rocket_data[rocket_name]["energy_joules"], # For crater calculation
        'vx_global': np.array(global_vx),
        'vy_global': np.array(global_vy),
        'vz_global': np.array(global_vz),
        'ax_global': np.array(global_ax),
        'ay_global': np.array(global_ay),
        'az_global': np.array(global_az),
    }
    
    # Initialize plots for animation (use global coordinates)
    line, = ax.plot([], [], [], '-', color=params["color"], linewidth=2, label=f'{rocket_name} Trajectory')
    point, = ax.plot([], [], [], 'o', color=params["color"], markersize=5)
    trajectory_lines[rocket_name] = line
    trajectory_points[rocket_name] = point

    # Initialize velocity and acceleration arrows
    # We'll plot a few arrows along the path.
    velocity_arrows[rocket_name] = []
    acceleration_arrows[rocket_name] = []
    for _ in range(3): # Plot 3 arrows per rocket
        v_arrow = ax.quiver([], [], [], [], [], [], color='green', length=ARROW_LENGTH_SCALE, arrow_length_ratio=0.3)
        a_arrow = ax.quiver([], [], [], [], [], [], color='purple', length=ARROW_LENGTH_SCALE, arrow_length_ratio=0.3)
        velocity_arrows[rocket_name].append(v_arrow)
        acceleration_arrows[rocket_name].append(a_arrow)

    # Find collision points with atmospheric layers (need to convert these to global XYZ too)
    collision_points_for_rocket = []
    for layer_name, layer_params in ATMOSPHERE_LAYERS.items():
        min_alt = layer_params["min_alt"]
        max_alt = layer_params["max_alt"]

        # Check for entry into layer (from below min_alt to above min_alt)
        entry_indices = np.where((y_traj_2d[:-1] < min_alt) & (y_traj_2d[1:] >= min_alt))[0]
        for idx in entry_indices:
            # Interpolate to find precise intersection point in local coords
            if y_traj_2d[idx+1] - y_traj_2d[idx] != 0:
                fraction = (min_alt - y_traj_2d[idx]) / (y_traj_2d[idx+1] - y_traj_2d[idx])
                coll_local_x_range = x_traj_2d[idx] + fraction * (x_traj_2d[idx+1] - x_traj_2d[idx])
                coll_local_y_alt = min_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                local_x_disp = coll_local_x_range * np.sin(launch_azimuth_rad)
                local_y_disp = coll_local_x_range * np.cos(launch_azimuth_rad)
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(local_x_disp, local_y_disp, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                
                # Get velocity vector at this point in global Cartesian coordinates
                # Find the index in the original trajectory data that is closest to this collision point
                closest_idx_in_traj = np.argmin(np.abs(x_traj_2d - coll_local_x_range))
                v_at_collision_x = global_vx[closest_idx_in_traj]
                v_at_collision_y = global_vy[closest_idx_in_traj]
                v_at_collision_z = global_vz[closest_idx_in_traj]
                velocity_vector_at_collision = np.array([v_at_collision_x, v_at_collision_y, v_at_collision_z])

                # Calculate normal vector (from Earth's center to collision point)
                # Normal vector points outwards from the spherical layer.
                r_collision = R_EARTH + coll_global_alt
                normal_vector_at_collision = np.array([coll_global_X, coll_global_Y, coll_global_Z]) / r_collision
                
                # Calculate flux (dot product of velocity and normal vector)
                flux_value = np.dot(velocity_vector_at_collision, normal_vector_at_collision)

                # Store flux value, velocity vector, and normal vector
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Entry {layer_name}', coll_global_lat, coll_global_lon, coll_global_alt, flux_value, velocity_vector_at_collision.tolist(), normal_vector_at_collision.tolist()))
        
        # Check for exit from layer (from below max_alt to above max_alt)
        exit_indices = np.where((y_traj_2d[:-1] < max_alt) & (y_traj_2d[1:] >= max_alt))[0]
        for idx in exit_indices:
            if y_traj_2d[idx+1] - y_traj_2d[idx] != 0:
                fraction = (max_alt - y_traj_2d[idx]) / (y_traj_2d[idx+1] - y_traj_2d[idx])
                coll_local_x_range = x_traj_2d[idx] + fraction * (x_traj_2d[idx+1] - x_traj_2d[idx])
                coll_local_y_alt = max_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                local_x_disp = coll_local_x_range * np.sin(launch_azimuth_rad)
                local_y_disp = coll_local_x_range * np.cos(launch_azimuth_rad)
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(local_x_disp, local_y_disp, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                
                # For exit points, we don't calculate flux as per the request, but store placeholders
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Exit {layer_name}', coll_global_lat, coll_global_lon, coll_global_alt, None, None, None))

    collision_markers[rocket_name] = ax.plot([], [], [], 'X', color=params["color"], markersize=8, label=f'{rocket_name} Collisions')[0]
    trajectory_data[rocket_name]['collisions'] = collision_points_for_rocket


    # Calculate crater dimensions for the final impact point
    crater_radius, crater_depth = calculate_crater_dimensions(
        rocket_data[rocket_name]["energy_joules"], DEFAULT_MATERIAL_STRENGTH_FOR_SORTING
    )
    trajectory_data[rocket_name]['crater_radius'] = crater_radius
    trajectory_data[rocket_name]['crater_depth'] = crater_depth
    
    # Initialize crater plot (as a sphere for simplicity)
    # The crater is centered at the impact point, with its bottom at alt 0.
    impact_x, impact_y, impact_z = trajectory_data[rocket_name]['x_global'][-1], trajectory_data[rocket_name]['y_global'][-1], trajectory_data[rocket_name]['z_global'][-1]
    
    # To plot a sphere at the impact point, we need its center and radius
    # The center of the sphere will be at (impact_x, impact_y, impact_z + crater_radius)
    # The radius of the sphere is the crater_radius
    crater_sphere_center_x = impact_x
    crater_sphere_center_y = impact_y
    crater_sphere_center_z = impact_z + crater_radius # Center of sphere is above ground by its radius

    # Create sphere data for crater
    crater_u = np.linspace(0, 2 * np.pi, 20)
    crater_v = np.linspace(0, np.pi, 10)
    crater_X = crater_radius * np.outer(np.cos(crater_u), np.sin(crater_v)) + crater_sphere_center_x
    crater_Y = crater_radius * np.outer(np.sin(crater_u), np.sin(crater_v)) + crater_sphere_center_y
    crater_Z = crater_radius * np.outer(np.ones(np.size(crater_u)), np.cos(crater_v)) + crater_sphere_center_z
    
    # Store crater plot for animation
    crater_plots[rocket_name] = ax.plot_surface(crater_X, crater_Y, crater_Z, color='brown', alpha=0.6, label=f'{rocket_name} Crater')
    crater_plots[rocket_name].set_visible(False) # Hide initially


# Set plot limits to encompass the Earth sphere and max altitude
ax.set_xlim(-R_EARTH * 1.2, R_EARTH * 1.2)
ax.set_ylim(-R_EARTH * 1.2, R_EARTH * 1.2)
ax.set_zlim(-R_EARTH * 1.2, R_EARTH * 1.2)
ax.set_box_aspect([1,1,1]) # Equal aspect ratio for a sphere

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Rocket Trajectories and Atmospheric Layers over Earth (Matplotlib)')
ax.legend()

def init_animation():
    for rocket_name in missiles_to_simulate:
        if rocket_name in trajectory_lines:
            trajectory_lines[rocket_name].set_data_3d([], [], [])
            trajectory_points[rocket_name].set_data_3d([], [], [])
            collision_markers[rocket_name].set_data_3d([], [], [])
            crater_plots[rocket_name].set_visible(False) # Hide craters
            # Remove existing arrows from previous frames if any
            for arrow in velocity_arrows[rocket_name]:
                arrow.remove()
            for arrow in acceleration_arrows[rocket_name]:
                arrow.remove()
            velocity_arrows[rocket_name].clear()
            acceleration_arrows[rocket_name].clear()
            # Clear flux markers
            if rocket_name in flux_markers:
                for marker in flux_markers[rocket_name]:
                    marker.remove()
                flux_markers[rocket_name].clear()

    satellite_plot.set_data_3d([], [], []) # Initialize satellite
    
    # Return all artists that will be updated
    all_artists = list(trajectory_lines.values()) + \
                  list(trajectory_points.values()) + \
                  list(collision_markers.values()) + \
                  list(crater_plots.values()) + \
                  [satellite_plot]
    # Add all initial (empty) velocity and acceleration arrows
    for rocket_name in missiles_to_simulate:
        for arrow in velocity_arrows[rocket_name]:
            all_artists.append(arrow)
        for arrow in acceleration_arrows[rocket_name]:
            all_artists.append(arrow)
    return all_artists

def update_animation(frame):
    artists = []
    for rocket_name, data in trajectory_data.items():
        num_points = data['length']
        current_idx = min(frame, num_points - 1) # Ensure index doesn't exceed bounds

        # Update trajectory line (use global coordinates)
        trajectory_lines[rocket_name].set_data_3d(data['x_global'][:current_idx+1], data['y_global'][:current_idx+1], data['z_global'][:current_idx+1])
        artists.append(trajectory_lines[rocket_name])

        # Update rocket point (use global coordinates)
        trajectory_points[rocket_name].set_data_3d([data['x_global'][current_idx]], [data['y_global'][current_idx]], [data['z_global'][current_idx]])
        artists.append(trajectory_points[rocket_name])

        # Update collision markers (show all up to current frame)
        coll_x_display = []
        coll_y_display = []
        coll_z_display = []
        
        # Clear previous flux markers for this rocket
        if rocket_name in flux_markers:
            for marker in flux_markers[rocket_name]:
                marker.remove()
            flux_markers[rocket_name].clear()
        else:
            flux_markers[rocket_name] = []

        for coll_global_x, coll_global_y, coll_global_z, coll_type, coll_global_lat, coll_global_lon, coll_global_alt, flux_val, vel_vec, norm_vec in data['collisions']:
            # Check if the rocket has passed the collision point in terms of distance from launch
            current_rocket_pos = np.array([data['x_global'][current_idx], data['y_global'][current_idx], data['z_global'][current_idx]])
            collision_point_pos = np.array([coll_global_x, coll_global_y, coll_global_z])
            
            if np.linalg.norm(current_rocket_pos - convert_lat_lon_alt_to_cartesian(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, 0)) > \
               np.linalg.norm(collision_point_pos - convert_lat_lon_alt_to_cartesian(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, 0)):
                coll_x_display.append(coll_global_x)
                coll_y_display.append(coll_global_y)
                coll_z_display.append(coll_global_z)

                # If it's an entry point, plot velocity, normal, and flux
                if "Entry" in coll_type and flux_val is not None and vel_vec is not None and norm_vec is not None:
                    # Plot velocity vector
                    v_arrow = ax.quiver(coll_global_x, coll_global_y, coll_global_z,
                                        vel_vec[0], vel_vec[1], vel_vec[2],
                                        color='lime', length=ARROW_LENGTH_SCALE / 2, arrow_length_ratio=0.3, label='Velocity Vector')
                    artists.append(v_arrow)
                    
                    # Plot normal vector
                    n_arrow = ax.quiver(coll_global_x, coll_global_y, coll_global_z,
                                        norm_vec[0], norm_vec[1], norm_vec[2],
                                        color='blue', length=ARROW_LENGTH_SCALE / 4, arrow_length_ratio=0.3, label='Normal Vector')
                    artists.append(n_arrow)

                    # Plot flux magnitude as a scaled marker
                    # Scale flux value to a reasonable marker size
                    flux_marker_size = 5 + abs(flux_val) / 1000 # Adjust scaling for visibility
                    flux_marker, = ax.plot([coll_global_x], [coll_global_y], [coll_global_z],
                                            'o', color='yellow', markersize=flux_marker_size, alpha=0.7,
                                            label=f'Flux: {flux_val:.2f}')
                    flux_markers[rocket_name].append(flux_marker)
                    artists.append(flux_marker)


        collision_markers[rocket_name].set_data_3d(coll_x_display, coll_y_display, coll_z_display)
        artists.append(collision_markers[rocket_name])

        # Show crater at the very end of the animation (when rocket has impacted)
        if current_idx == num_points - 1:
            crater_plots[rocket_name].set_visible(True)
            artists.append(crater_plots[rocket_name])
        else:
            crater_plots[rocket_name].set_visible(False)


        # Update velocity and acceleration arrows for the rocket's path
        # Clear previous arrows
        for arrow in velocity_arrows[rocket_name]:
            arrow.remove()
        for arrow in acceleration_arrows[rocket_name]:
            arrow.remove()
        velocity_arrows[rocket_name].clear()
        acceleration_arrows[rocket_name].clear()

        # Plot new arrows at a few points along the current visible trajectory
        # Ensure we don't try to plot arrows for a trajectory that hasn't started yet
        if current_idx > 0:
            arrow_indices = np.linspace(0, current_idx, min(3, current_idx + 1), dtype=int)
            for idx in arrow_indices:
                # Velocity arrow
                v_origin = [data['x_global'][idx], data['y_global'][idx], data['z_global'][idx]]
                v_direction = [data['vx_global'][idx], data['vy_global'][idx], data['vz_global'][idx]]
                v_mag = np.linalg.norm(v_direction)
                if v_mag > 0:
                    v_scaled_direction = np.array(v_direction) / v_mag * ARROW_LENGTH_SCALE
                    v_arrow = ax.quiver(v_origin[0], v_origin[1], v_origin[2],
                                        v_scaled_direction[0], v_scaled_direction[1], v_scaled_direction[2],
                                        color='green', length=1, arrow_length_ratio=0.3)
                    velocity_arrows[rocket_name].append(v_arrow)
                    artists.append(v_arrow)

                # Acceleration arrow
                a_origin = [data['x_global'][idx], data['y_global'][idx], data['z_global'][idx]]
                a_direction = [data['ax_global'][idx], data['ay_global'][idx], data['az_global'][idx]]
                a_mag = np.linalg.norm(a_direction)
                if a_mag > 0:
                    a_scaled_direction = np.array(a_direction) / a_mag * ARROW_LENGTH_SCALE
                    a_arrow = ax.quiver(a_origin[0], a_origin[1], a_origin[2],
                                        a_scaled_direction[0], a_scaled_direction[1], a_scaled_direction[2],
                                        color='purple', length=1, arrow_length_ratio=0.3)
                    acceleration_arrows[rocket_name].append(a_arrow)
                    artists.append(a_arrow)


    # Animate satellite position (simple orbit around Earth)
    angle = frame * 0.01 # Adjust speed of satellite rotation
    sat_lat_curr = sat_lat_init
    sat_lon_curr = sat_lon_init + np.degrees(angle) # Rotate longitude
    sat_x_curr, sat_y_curr, sat_z_curr = convert_lat_lon_alt_to_cartesian(sat_lat_curr, sat_lon_curr, satellite_alt_m)
    satellite_plot.set_data_3d([sat_x_curr], [sat_y_curr], [sat_z_curr])
    artists.append(satellite_plot)


    # Rotate camera for better view
    ax.view_init(elev=20, azim=frame * 0.5) # Slower rotation

    return artists

# Determine total frames for animation
max_traj_length = max(data['length'] for data in trajectory_data.values()) if trajectory_data else 100
total_frames = max_traj_length # Animate until the longest trajectory is complete

anim = FuncAnimation(fig, update_animation, frames=total_frames,
                     init_func=init_animation, blit=False, interval=ANIMATION_INTERVAL_MS, repeat=False)

plt.show()

# =============================================================================
# 6. MCMC Simulation for Impact Probability and Failure Rate
# =============================================================================

# Calculate the average final impact point from the simulated rockets (using local coordinates)
all_final_x_local = [data['final_impact_x_local'] for data in trajectory_data.values() if 'final_impact_x_local' in data]
all_final_y_local = [data['final_impact_y_local'] for data in trajectory_data.values() if 'final_impact_y_local' in data]

if all_final_x_local and all_final_y_local:
    mcmc_target_center_x = np.mean(all_final_x_local)
    mcmc_target_center_y = np.mean(all_final_y_local)
else:
    # Default to center of Nepal map if no rockets were simulated or trajectory data missing
    mcmc_target_center_x = NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2
    mcmc_target_center_y = NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2

# Parameters for the MCMC target distribution (2D Gaussian)
# Let's assume a standard deviation of 2 km (2000 meters) for impact precision.
target_mu = np.array([mcmc_target_center_x, mcmc_target_center_y])
target_cov = np.diag([2000**2, 2000**2]) # Covariance matrix for 2km std dev in x and y

# Define the target probability density function (PDF)
target_pdf = multivariate_normal(mean=target_mu, cov=target_cov)

# MCMC Parameters
num_samples = 20000
burn_in = 5000 # Discard initial samples
proposal_std = 500 # Standard deviation for the proposal distribution (in meters)

# MCMC Simulation (Metropolis-Hastings)
samples = []
current_sample = target_mu # Start at the mean of the target distribution

for i in range(num_samples + burn_in):
    # Propose a new sample
    proposal = current_sample + np.random.normal(0, proposal_std, size=2)

    # Calculate acceptance ratio
    prob_current = target_pdf.pdf(current_sample)
    prob_proposal = target_pdf.pdf(proposal)

    if prob_current == 0 and prob_proposal == 0: # Avoid division by zero if both are zero
        alpha = 1 # Treat as always accepted if outside high probability region
    elif prob_current == 0: # Current is zero, proposal is non-zero, always accept (to move towards higher density)
        alpha = 1
    else:
        alpha = min(1, prob_proposal / prob_current)

    # Accept or reject the proposal
    if np.random.rand() < alpha:
        current_sample = proposal
    
    if i >= burn_in:
        samples.append(current_sample)

samples_array = np.array(samples)

# Calculate Failure Rate for a 50 km^2 Area
TARGET_AREA_KM2 = 50
TARGET_AREA_M2 = TARGET_AREA_KM2 * 1e6 # Convert to square meters
TARGET_RADIUS_M = np.sqrt(TARGET_AREA_M2 / np.pi) # Radius of a circle with 50 km^2 area

# Calculate distances from the target center for each sample
distances_from_target = np.linalg.norm(samples_array - target_mu, axis=1)

# Count samples outside the target radius
failures = np.sum(distances_from_target > TARGET_RADIUS_M)
failure_rate_percent = (failures / len(samples_array)) * 100

print(f"\n--- MCMC Simulation Results ---")
print(f"Total MCMC samples: {len(samples_array)}")
print(f"Target area: {TARGET_AREA_KM2} km^2 (approx. radius {TARGET_RADIUS_M/1000:.2f} km)")
print(f"Number of samples outside target area (failures): {failures}")
print(f"Calculated Failure Rate: {failure_rate_percent:.2f}%")


# Plotting MCMC Simulation on Matplotlib (2D map)
fig_mcmc = plt.figure(figsize=(10, 8))
ax_mcmc = fig_mcmc.add_subplot(111)

# Plot Nepal ground plane (2D representation for MCMC) using hardcoded boundary
nepal_boundary_local_x, nepal_boundary_local_y = [], []
# Assuming the MCMC plot origin is NEPAL_LON_MIN, NEPAL_LAT_MIN
mcmc_origin_lat = NEPAL_LAT_MIN
mcmc_origin_lon = NEPAL_LON_MIN

for lon, lat in NEPAL_BOUNDARY_COORDS_LON_LAT:
    local_x, local_y = convert_global_lat_lon_to_local_meters(lat, lon, mcmc_origin_lat, mcmc_origin_lon)
    nepal_boundary_local_x.append(local_x)
    nepal_boundary_local_y.append(local_y)

ax_mcmc.plot(nepal_boundary_local_x, nepal_boundary_local_y, color='green', linewidth=2, label='Nepal Region (2D)')
ax_mcmc.fill(nepal_boundary_local_x, nepal_boundary_local_y, color='green', alpha=0.2)


# Plot MCMC samples
ax_mcmc.scatter(samples_array[:, 0], samples_array[:, 1], s=1, alpha=0.1, color='red', label='MCMC Impact Samples')

# Plot the target center
ax_mcmc.plot(target_mu[0], target_mu[1], 'x', color='blue', markersize=10, label='MCMC Target Center')

# Plot the 50 km^2 target area (circle)
circle = plt.Circle((target_mu[0], target_mu[1]), TARGET_RADIUS_M, color='blue', fill=False, linestyle='--', label=f'{TARGET_AREA_KM2} km\u00b2 Target Area')
ax_mcmc.add_patch(circle)
ax_mcmc.set_aspect('equal', adjustable='box') # Ensure circle looks like a circle

# Set plot limits
ax_mcmc.set_xlim(NEPAL_X_MIN, NEPAL_X_MAX)
ax_mcmc.set_ylim(NEPAL_Y_MIN, NEPAL_Y_MAX)

ax_mcmc.set_xlabel('X Coordinate (meters)')
ax_mcmc.set_ylabel('Y Coordinate (meters)')
ax_mcmc.set_title('MCMC Simulation of Rocket Impact Points over Nepal (2D Map)')

# Display failure rate on the plot
ax_mcmc.text(NEPAL_X_MIN + 0.02*(NEPAL_X_MAX-NEPAL_X_MIN), NEPAL_Y_MAX - 0.05*(NEPAL_Y_MAX-NEPAL_Y_MIN),
             f'Failure Rate for {TARGET_AREA_KM2} km\u00b2 Area: {failure_rate_percent:.2f}%',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax_mcmc.legend()
plt.show()

# =============================================================================
# 5. PyDeck Visualization
# =============================================================================

def convert_meters_to_lat_lon(x_meters, y_meters):
    """Converts meters from Nepal's origin (local flat map) to approximate latitude and longitude."""
    # This function is used for the 2D MCMC plot and PyDeck's interpretation of local coordinates.
    # For PyDeck, it effectively maps the local X/Y offsets to lat/lon offsets from NEPAL_LON_MIN, NEPAL_LAT_MIN.
    lon = NEPAL_LON_MIN + (x_meters / (KM_PER_DEG_LON_AT_NEPAL_AVG_LAT * 1000))
    lat = NEPAL_LAT_MIN + (y_meters / (KM_PER_DEG_LAT * 1000))
    return lat, lon

def visualize_with_pydeck(trajectory_data_pydeck, collision_markers_data_pydeck, mcmc_results_pydeck, target_radius_pydeck):
    # Base map view centered over Nepal
    view_state = pdk.ViewState(
        latitude=(NEPAL_LAT_MIN + NEPAL_LAT_MAX) / 2,
        longitude=(NEPAL_LON_MIN + NEPAL_LON_MAX) / 2,
        zoom=5,
        pitch=45,
        bearing=0
    )

    layers = []

    # 1. Nepal Ground Plane (using hardcoded boundary)
    nepal_polygon_coords_pydeck = [[lon, lat] for lon, lat in NEPAL_BOUNDARY_COORDS_LON_LAT]
    ground_df = pd.DataFrame({
        'polygon': [nepal_polygon_coords_pydeck],
        'color': [[0, 128, 0, 180]] # Green with some transparency
    })
    layers.append(
        pdk.Layer(
            "PolygonLayer",
            ground_df,
            get_polygon="polygon",
            filled=True,
            get_fill_color="color",
            stroked=False,
            get_line_color=[0, 0, 0],
            line_width_min_pixels=1,
            pickable=True,
            auto_highlight=True,
        )
    )

    # 2. Atmospheric Layers (as top and bottom polygons with altitude)
    for layer_name, params in ATMOSPHERE_LAYERS.items():
        min_alt = params["min_alt"]
        max_alt = params["max_alt"]
        
        # Convert matplotlib color string to RGBA (0-255)
        rgba_color_mpl = matplotlib.colors.to_rgba(params["color"])
        color_rgba_pydeck = [int(c * 255) for c in rgba_color_mpl[:3]] + [int(params["alpha"] * 255)]

        # Define the corners of the atmospheric layer polygons
        # Bottom surface
        bottom_polygon_coords = [
            [NEPAL_LON_MIN, NEPAL_LAT_MIN, min_alt],
            [NEPAL_LON_MAX, NEPAL_LAT_MIN, min_alt],
            [NEPAL_LON_MAX, NEPAL_LAT_MAX, min_alt],
            [NEPAL_LON_MIN, NEPAL_LAT_MAX, min_alt],
            [NEPAL_LON_MIN, NEPAL_LAT_MIN, min_alt]
        ]
        
        # Top surface
        top_polygon_coords = [
            [NEPAL_LON_MIN, NEPAL_LAT_MIN, max_alt],
            [NEPAL_LON_MAX, NEPAL_LAT_MIN, max_alt],
            [NEPAL_LON_MAX, NEPAL_LAT_MAX, max_alt],
            [NEPAL_LON_MIN, NEPAL_LAT_MAX, max_alt],
            [NEPAL_LON_MIN, NEPAL_LAT_MIN, max_alt]
        ]

        layer_df = pd.DataFrame({
            'polygon': [bottom_polygon_coords, top_polygon_coords],
            'color': [color_rgba_pydeck, color_rgba_pydeck],
            'name': [f"{layer_name} Bottom", f"{layer_name} Top"]
        })
        
        layers.append(
            pdk.Layer(
                "PolygonLayer",
                layer_df,
                get_polygon="polygon",
                filled=True,
                get_fill_color="color",
                stroked=False,
                get_line_color=[0, 0, 0],
                line_width_min_pixels=1,
                pickable=True,
                auto_highlight=True,
            )
        )

    # 3. Rocket Trajectories
    for rocket_name, data in trajectory_data_pydeck.items():
        # Use global coordinates for PyDeck path
        path_coords = []
        # We need to convert the global Cartesian coordinates back to lat/lon/alt for PyDeck's PathLayer
        # This requires a helper function for Cartesian to Spherical.
        def convert_cartesian_to_lat_lon_alt(x, y, z):
            r = np.sqrt(x**2 + y**2 + z**2)
            lat_rad = np.arcsin(z / r)
            lon_rad = np.arctan2(y, x)
            alt_m = r - R_EARTH
            return np.degrees(lat_rad), np.degrees(lon_rad), alt_m

        for i in range(len(data['x_global'])):
            lat, lon, alt = convert_cartesian_to_lat_lon_alt(data['x_global'][i], data['y_global'][i], data['z_global'][i])
            path_coords.append([lon, lat, alt])
        
        # Convert named color string to RGBA (0-255)
        rgba_color_mpl = matplotlib.colors.to_rgba(data['color'])
        color_pydeck = [int(c * 255) for c in rgba_color_mpl]
        
        path_df = pd.DataFrame({
            'path': [path_coords],
            'color': [color_pydeck],
            'name': [rocket_name]
        })
        
        layers.append(
            pdk.Layer(
                "PathLayer",
                path_df,
                get_path="path",
                get_color="color",
                width_scale=5,
                width_min_pixels=2,
                pickable=True,
                auto_highlight=True,
            )
        )

    # 4. Collision Markers and Flux Visualization
    all_collision_points_pydeck = []
    all_velocity_vectors_pydeck = []
    all_normal_vectors_pydeck = []
    all_flux_markers_pydeck = []

    for rocket_name, collisions in collision_markers_data_pydeck.items():
        for coll_global_x, coll_global_y, coll_global_z, coll_type, coll_global_lat, coll_global_lon, coll_global_alt, flux_val, vel_vec, norm_vec in collisions:
            # Main collision marker
            all_collision_points_pydeck.append({
                'position': [coll_global_lon, coll_global_lat, coll_global_alt],
                'color': [255, 0, 0, 255], # Red 'X'
                'size': 100, # Size in meters
                'name': f"{rocket_name} {coll_type}"
            })

            if "Entry" in coll_type and flux_val is not None and vel_vec is not None and norm_vec is not None:
                # Velocity vector
                all_velocity_vectors_pydeck.append({
                    'source_position': [coll_global_lon, coll_global_lat, coll_global_alt],
                    'target_position': [coll_global_lon + vel_vec[0]/ARROW_LENGTH_SCALE, coll_global_lat + vel_vec[1]/ARROW_LENGTH_SCALE, coll_global_alt + vel_vec[2]/ARROW_LENGTH_SCALE], # Scale vector for visualization
                    'color': [0, 255, 0, 255], # Lime green
                    'name': f"{rocket_name} Velocity"
                })
                # Normal vector
                all_normal_vectors_pydeck.append({
                    'source_position': [coll_global_lon, coll_global_lat, coll_global_alt],
                    'target_position': [coll_global_lon + norm_vec[0]/ARROW_LENGTH_SCALE, coll_global_lat + norm_vec[1]/ARROW_LENGTH_SCALE, coll_global_alt + norm_vec[2]/ARROW_LENGTH_SCALE], # Scale vector
                    'color': [0, 0, 255, 255], # Blue
                    'name': f"{rocket_name} Normal"
                })
                # Flux marker (scaled by flux magnitude)
                flux_radius = max(10, abs(flux_val) / 100) # Scale flux to a visible radius, min 10m
                all_flux_markers_pydeck.append({
                    'position': [coll_global_lon, coll_global_lat, coll_global_alt + flux_radius], # Place slightly above collision marker
                    'radius': flux_radius,
                    'color': [255, 255, 0, 150], # Yellow with transparency
                    'name': f"Flux: {flux_val:.2f}"
                })
    
    if all_collision_points_pydeck:
        collision_df = pd.DataFrame(all_collision_points_pydeck)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                collision_df,
                get_position="position",
                get_fill_color="color",
                get_radius="size",
                radius_scale=1,
                pickable=True,
                auto_highlight=True,
            )
        )
    if all_velocity_vectors_pydeck:
        vel_vec_df = pd.DataFrame(all_velocity_vectors_pydeck)
        layers.append(
            pdk.Layer(
                "ArcLayer",
                vel_vec_df,
                get_source_position="source_position",
                get_target_position="target_position",
                get_source_color="color",
                get_target_color="color",
                get_width=5,
                pickable=True,
                auto_highlight=True,
            )
        )
    if all_normal_vectors_pydeck:
        norm_vec_df = pd.DataFrame(all_normal_vectors_pydeck)
        layers.append(
            pdk.Layer(
                "ArcLayer",
                norm_vec_df,
                get_source_position="source_position",
                get_target_position="target_position",
                get_source_color="color",
                get_target_color="color",
                get_width=5,
                pickable=True,
                auto_highlight=True,
            )
        )
    if all_flux_markers_pydeck:
        flux_marker_df = pd.DataFrame(all_flux_markers_pydeck)
        layers.append(
            pdk.Layer(
                "SphereLayer",
                flux_marker_df,
                get_position="position",
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
        )


    # Add Satellite to PyDeck (as a single point)
    sat_lat_pydeck, sat_lon_pydeck, sat_alt_pydeck = convert_local_meters_to_global_lat_lon_alt(0, 0, satellite_alt_m, sat_lat_init, sat_lon_init)
    satellite_pydeck_df = pd.DataFrame([{
        'position': [sat_lon_pydeck, sat_lat_pydeck, sat_alt_pydeck],
        'color': [0, 255, 255, 255], # Cyan
        'size': 5000, # Larger size for visibility
        'name': 'Satellite'
    }])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            satellite_pydeck_df,
            get_position="position",
            get_fill_color="color",
            get_radius="size",
            radius_scale=1,
            pickable=True,
            auto_highlight=True,
        )
    )

    # Add Explosion Craters to PyDeck
    crater_data_pydeck = []
    for rocket_name, data in trajectory_data_pydeck.items():
        impact_lat = data['final_impact_global_lat']
        impact_lon = data['final_impact_global_lon']
        crater_radius = data['crater_radius']
        crater_depth = data['crater_depth'] # Not directly used by SphereLayer, but good for context

        # For SphereLayer, the position is the center of the sphere.
        # We want the bottom of the sphere to be at ground level (altitude 0).
        # So, the center's altitude should be the radius.
        crater_center_alt = crater_radius

        crater_data_pydeck.append({
            'position': [impact_lon, impact_lat, crater_center_alt],
            'radius': crater_radius,
            'color': [139, 69, 19, 150], # Brown with transparency
            'name': f"{rocket_name} Crater (R: {crater_radius:.1f}m)"
        })
    
    if crater_data_pydeck:
        crater_df_pydeck = pd.DataFrame(crater_data_pydeck)
        layers.append(
            pdk.Layer(
                "SphereLayer",
                crater_df_pydeck,
                get_position="position",
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                auto_highlight=True,
            )
        )

    # Add MCMC samples to PyDeck (as small circles on the ground)
    mcmc_points_pydeck = []
    mcmc_target_center_lat, mcmc_target_center_lon, _ = convert_local_meters_to_global_lat_lon_alt(
        mcmc_results_pydeck['target_mu'][0] - (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2),
        mcmc_results_pydeck['target_mu'][1] - (NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2),
        0, NEPAL_LAT_CENTER, NEPAL_LON_CENTER
    )

    for sample_x_local, sample_y_local in mcmc_results_pydeck['samples_array']:
        # Convert local MCMC coordinates back to global lat/lon
        # Adjusting for the origin of the local MCMC map relative to Nepal's center
        relative_x = sample_x_local - (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2)
        relative_y = sample_y_local - (NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2)
        
        lat, lon, _ = convert_local_meters_to_global_lat_lon_alt(relative_x, relative_y, 0, NEPAL_LAT_CENTER, NEPAL_LON_CENTER)
        mcmc_points_pydeck.append({
            'position': [lon, lat, 0], # Altitude 0 for ground impacts
            'color': [255, 0, 0, 10], # Red with high transparency
            'radius': 50 # Small radius for individual points
        })
    
    if mcmc_points_pydeck:
        mcmc_df_pydeck = pd.DataFrame(mcmc_points_pydeck)
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                mcmc_df_pydeck,
                get_position="position",
                get_fill_color="color",
                get_radius="radius",
                pickable=True,
                auto_highlight=True,
            )
        )

    # Add MCMC target area circle
    target_circle_pydeck = pd.DataFrame([{
        'position': [mcmc_target_center_lon, mcmc_target_center_lat, 0],
        'radius': target_radius_pydeck,
        'color': [0, 0, 255, 100], # Blue with transparency
        'name': f"{TARGET_AREA_KM2} km² Target Area"
    }])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            target_circle_pydeck,
            get_position="position",
            get_fill_color="color",
            get_radius="radius",
            stroked=True,
            get_line_color=[0, 0, 255, 200],
            line_width_min_pixels=2,
            pickable=True,
            auto_highlight=True,
        )
    )


    # Render the deck
    deck = pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9", # Light map style
        initial_view_state=view_state,
        layers=layers,
        tooltip={"text": "{name}\nAltitude: {position.2}m"} # Tooltip for layers/points
    )

    # Save to HTML file. This file will be generated in the environment.
    deck.to_html("nepal_rocket_simulation_pydeck.html") 
    print("PyDeck visualization saved to nepal_rocket_simulation_pydeck.html. Please open this file in a web browser to view the interactive map.")


# =============================================================================
# 6. MCMC Simulation for Impact Probability and Failure Rate
# =============================================================================

# Calculate the average final impact point from the simulated rockets (using local coordinates)
all_final_x_local = [data['final_impact_x_local'] for data in trajectory_data.values() if 'final_impact_x_local' in data]
all_final_y_local = [data['final_impact_y_local'] for data in trajectory_data.values() if 'final_impact_y_local' in data]

if all_final_x_local and all_final_y_local:
    mcmc_target_center_x = np.mean(all_final_x_local)
    mcmc_target_center_y = np.mean(all_final_y_local)
else:
    # Default to center of Nepal map if no rockets were simulated or trajectory data missing
    mcmc_target_center_x = NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2
    mcmc_target_center_y = NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2

# Parameters for the MCMC target distribution (2D Gaussian)
# Let's assume a standard deviation of 2 km (2000 meters) for impact precision.
target_mu = np.array([mcmc_target_center_x, mcmc_target_center_y])
target_cov = np.diag([2000**2, 2000**2]) # Covariance matrix for 2km std dev in x and y

# Define the target probability density function (PDF)
target_pdf = multivariate_normal(mean=target_mu, cov=target_cov)

# MCMC Parameters
num_samples = 20000
burn_in = 5000 # Discard initial samples
proposal_std = 500 # Standard deviation for the proposal distribution (in meters)

# MCMC Simulation (Metropolis-Hastings)
samples = []
current_sample = target_mu # Start at the mean of the target distribution

for i in range(num_samples + burn_in):
    # Propose a new sample
    proposal = current_sample + np.random.normal(0, proposal_std, size=2)

    # Calculate acceptance ratio
    prob_current = target_pdf.pdf(current_sample)
    prob_proposal = target_pdf.pdf(proposal)

    if prob_current == 0 and prob_proposal == 0: # Avoid division by zero if both are zero
        alpha = 1 # Treat as always accepted if outside high probability region
    elif prob_current == 0: # Current is zero, proposal is non-zero, always accept (to move towards higher density)
        alpha = 1
    else:
        alpha = min(1, prob_proposal / prob_current)

    # Accept or reject the proposal
    if np.random.rand() < alpha:
        current_sample = proposal
    
    if i >= burn_in:
        samples.append(current_sample)

samples_array = np.array(samples)

# Calculate Failure Rate for a 50 km^2 Area
TARGET_AREA_KM2 = 50
TARGET_AREA_M2 = TARGET_AREA_KM2 * 1e6 # Convert to square meters
TARGET_RADIUS_M = np.sqrt(TARGET_AREA_M2 / np.pi) # Radius of a circle with 50 km^2 area

# Calculate distances from the target center for each sample
distances_from_target = np.linalg.norm(samples_array - target_mu, axis=1)

# Count samples outside the target radius
failures = np.sum(distances_from_target > TARGET_RADIUS_M)
failure_rate_percent = (failures / len(samples_array)) * 100

print(f"\n--- MCMC Simulation Results ---")
print(f"Total MCMC samples: {len(samples_array)}")
print(f"Target area: {TARGET_AREA_KM2} km^2 (approx. radius {TARGET_RADIUS_M/1000:.2f} km)")
print(f"Number of samples outside target area (failures): {failures}")
print(f"Calculated Failure Rate: {failure_rate_percent:.2f}%")


# Plotting MCMC Simulation on Matplotlib (2D map)
fig_mcmc = plt.figure(figsize=(10, 8))
ax_mcmc = fig_mcmc.add_subplot(111)

# Plot Nepal ground plane (2D representation for MCMC) using hardcoded boundary
nepal_boundary_local_x, nepal_boundary_local_y = [], []
# Assuming the MCMC plot origin is NEPAL_LON_MIN, NEPAL_LAT_MIN
mcmc_origin_lat = NEPAL_LAT_MIN
mcmc_origin_lon = NEPAL_LON_MIN

for lon, lat in NEPAL_BOUNDARY_COORDS_LON_LAT:
    local_x, local_y = convert_global_lat_lon_to_local_meters(lat, lon, mcmc_origin_lat, mcmc_origin_lon)
    nepal_boundary_local_x.append(local_x)
    nepal_boundary_local_y.append(local_y)

ax_mcmc.plot(nepal_boundary_local_x, nepal_boundary_local_y, color='green', linewidth=2, label='Nepal Region (2D)')
ax_mcmc.fill(nepal_boundary_local_x, nepal_boundary_local_y, color='green', alpha=0.2)


# Plot MCMC samples
ax_mcmc.scatter(samples_array[:, 0], samples_array[:, 1], s=1, alpha=0.1, color='red', label='MCMC Impact Samples')

# Plot the target center
ax_mcmc.plot(target_mu[0], target_mu[1], 'x', color='blue', markersize=10, label='MCMC Target Center')

# Plot the 50 km^2 target area (circle)
circle = plt.Circle((target_mu[0], target_mu[1]), TARGET_RADIUS_M, color='blue', fill=False, linestyle='--', label=f'{TARGET_AREA_KM2} km\u00b2 Target Area')
ax_mcmc.add_patch(circle)
ax_mcmc.set_aspect('equal', adjustable='box') # Ensure circle looks like a circle

# Set plot limits
ax_mcmc.set_xlim(NEPAL_X_MIN, NEPAL_X_MAX)
ax_mcmc.set_ylim(NEPAL_Y_MIN, NEPAL_Y_MAX)

ax_mcmc.set_xlabel('X Coordinate (meters)')
ax_mcmc.set_ylabel('Y Coordinate (meters)')
ax_mcmc.set_title('MCMC Simulation of Rocket Impact Points over Nepal (2D Map)')

# Display failure rate on the plot
ax_mcmc.text(NEPAL_X_MIN + 0.02*(NEPAL_X_MAX-NEPAL_X_MIN), NEPAL_Y_MAX - 0.05*(NEPAL_Y_MAX-NEPAL_Y_MIN),
             f'Failure Rate for {TARGET_AREA_KM2} km\u00b2 Area: {failure_rate_percent:.2f}%',
             fontsize=12, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

ax_mcmc.legend()
plt.show()

# =============================================================================
# 7. Folium Visualization
# =============================================================================

def visualize_with_folium(trajectory_data_folium, mcmc_samples_folium, target_mu_folium, target_radius_folium, failure_rate_percent_folium):
    # Center the map over Nepal
    map_center = [NEPAL_LAT_CENTER, NEPAL_LON_CENTER]

    # Create Folium map with Satellite tiles
    m_satellite = folium.Map(location=map_center, zoom_start=6, tiles='Esri.WorldImagery')
    # Add classic OpenStreetMap tiles for switching
    folium.TileLayer('OpenStreetMap').add_to(m_satellite)

    # Add LayerControl to switch between base maps
    folium.LayerControl().add_to(m_satellite)

    # Add rocket trajectories
    for rocket_name, data in trajectory_data_folium.items():
        # Folium PolyLine expects [[lat1, lon1], [lat2, lon2], ...]
        path_coords = [[lat, lon] for lat, lon in zip(data['global_lats'], data['global_lons'])]
        folium.PolyLine(
            locations=path_coords,
            color=data['color'],
            weight=3,
            tooltip=f"{rocket_name} Trajectory"
        ).add_to(m_satellite)

        # Add collision markers and flux information
        for coll_global_x, coll_global_y, coll_global_z, coll_type, coll_global_lat, coll_global_lon, coll_global_alt, flux_val, vel_vec, norm_vec in data['collisions']:
            tooltip_text = f"{rocket_name} {coll_type} at {coll_global_alt/1000:.2f} km"
            if "Entry" in coll_type and flux_val is not None:
                tooltip_text += f"<br>Flux (V dot N): {flux_val:.2f}"
            
            folium.CircleMarker(
                location=[coll_global_lat, coll_global_lon],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                tooltip=tooltip_text
            ).add_to(m_satellite)

        # Add explosion crater marker at final impact point
        impact_lat = data['final_impact_global_lat']
        impact_lon = data['final_impact_global_lon']
        crater_radius_km = data['crater_radius'] / 1000 # Convert to km for tooltip
        folium.Circle(
            location=[impact_lat, impact_lon],
            radius=data['crater_radius'], # Folium radius is in meters
            color='brown',
            fill=True,
            fill_color='brown',
            fill_opacity=0.6,
            tooltip=f"{rocket_name} Crater (Radius: {crater_radius_km:.2f} km)"
        ).add_to(m_satellite)


    # Add MCMC samples
    mcmc_points_lat_lon = []
    # Convert MCMC target center from local meters to global lat/lon
    mcmc_target_center_lat, mcmc_target_center_lon, _ = convert_local_meters_to_global_lat_lon_alt(
        target_mu_folium[0] - (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2), # Adjust for local map origin
        target_mu_folium[1] - (NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2), # Adjust for local map origin
        0, NEPAL_LAT_CENTER, NEPAL_LON_CENTER
    )

    for sample_x_local, sample_y_local in mcmc_samples_folium:
        # Convert local MCMC coordinates back to global lat/lon
        relative_x = sample_x_local - (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2)
        relative_y = sample_y_local - (NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2)
        
        lat, lon, _ = convert_local_meters_to_global_lat_lon_alt(relative_x, relative_y, 0, NEPAL_LAT_CENTER, NEPAL_LON_CENTER)
        mcmc_points_lat_lon.append([lat, lon])

    # Convert MCMC samples to a DataFrame for easier handling with MarkerCluster or similar if needed
    mcmc_df = pd.DataFrame(mcmc_points_lat_lon, columns=['lat', 'lon'])

    # Add MCMC samples as small circles
    for idx, row in mcmc_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=1,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.1
        ).add_to(m_satellite)

    # Add MCMC target center
    folium.Marker(
        location=[mcmc_target_center_lat, mcmc_target_center_lon],
        icon=folium.Icon(color='blue', icon='info-sign'),
        tooltip="MCMC Target Center"
    ).add_to(m_satellite)

    # Add 50 km^2 target area circle
    # Folium Circle takes radius in meters
    folium.Circle(
        location=[mcmc_target_center_lat, mcmc_target_center_lon],
        radius=target_radius_folium,
        color='blue',
        fill=False,
        dash_array='5, 5',
        tooltip=f"{TARGET_AREA_KM2} km² Target Area"
    ).add_to(m_satellite)

    # Add failure rate text (as a simple HTML overlay or on a marker)
    folium.Marker(
        location=[NEPAL_LAT_MIN + 0.1, NEPAL_LON_MIN + 0.1], # Position text slightly offset
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black; background-color: white; padding: 5px; border-radius: 5px;">Failure Rate: {failure_rate_percent_folium:.2f}%</div>')
    ).add_to(m_satellite)


    # Save the satellite map
    m_satellite.save("folium_satellite_map.html")
    print("Folium satellite map saved to folium_satellite_map.html. Please open this file in a web browser.")

    # Create Folium map with Classic tiles (OpenStreetMap)
    m_classic = folium.Map(location=map_center, zoom_start=6, tiles='OpenStreetMap')
    # Add satellite tiles for switching
    folium.TileLayer('Esri.WorldImagery').add_to(m_classic)
    folium.LayerControl().add_to(m_classic)

    # Add rocket trajectories to classic map (same as satellite map)
    for rocket_name, data in trajectory_data_folium.items():
        path_coords = [[lat, lon] for lat, lon in zip(data['global_lats'], data['global_lons'])]
        folium.PolyLine(
            locations=path_coords,
            color=data['color'],
            weight=3,
            tooltip=f"{rocket_name} Trajectory"
        ).add_to(m_classic)
        for coll_global_x, coll_global_y, coll_global_z, coll_type, coll_global_lat, coll_global_lon, coll_global_alt, flux_val, vel_vec, norm_vec in data['collisions']:
            tooltip_text = f"{rocket_name} {coll_type} at {coll_global_alt/1000:.2f} km"
            if "Entry" in coll_type and flux_val is not None:
                tooltip_text += f"<br>Flux (V dot N): {flux_val:.2f}"
            folium.CircleMarker(
                location=[coll_global_lat, coll_global_lon],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                tooltip=tooltip_text
            ).add_to(m_classic)

        # Add explosion crater marker to classic map
        impact_lat = data['final_impact_global_lat']
        impact_lon = data['final_impact_global_lon']
        crater_radius_km = data['crater_radius'] / 1000
        folium.Circle(
            location=[impact_lat, impact_lon],
            radius=data['crater_radius'],
            color='brown',
            fill=True,
            fill_color='brown',
            fill_opacity=0.6,
            tooltip=f"{rocket_name} Crater (Radius: {crater_radius_km:.2f} km)"
        ).add_to(m_classic)

    # Add MCMC samples to classic map
    for idx, row in mcmc_df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=1,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=0.1
        ).add_to(m_classic)

    # Add MCMC target center to classic map
    folium.Marker(
        location=[mcmc_target_center_lat, mcmc_target_center_lon],
        icon=folium.Icon(color='blue', icon='info-sign'),
        tooltip="MCMC Target Center"
    ).add_to(m_classic)

    # Add 50 km^2 target area circle to classic map
    folium.Circle(
        location=[mcmc_target_center_lat, mcmc_target_center_lon],
        radius=target_radius_folium,
        color='blue',
        fill=False,
        dash_array='5, 5',
        tooltip=f"{TARGET_AREA_KM2} km² Target Area"
    ).add_to(m_classic)

    # Add failure rate text to classic map
    folium.Marker(
        location=[NEPAL_LAT_MIN + 0.1, NEPAL_LON_MIN + 0.1],
        icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: black; background-color: white; padding: 5px; border-radius: 5px;">Failure Rate: {failure_rate_percent_folium:.2f}%</div>')
    ).add_to(m_classic)

    # Save the classic map
    m_classic.save("folium_classic_map.html")
    print("Folium classic map saved to folium_classic_map.html. Please open this file in a web browser.")


# Main execution flow
# Pre-calculate trajectories and collision points for Matplotlib and PyDeck
# This block populates `trajectory_data` and `collision_markers`.

# --- MCMC Simulation for Impact Probability and Failure Rate (Moved here) ---
# Calculate the average final impact point from the simulated rockets (using local coordinates)
all_final_x_local = [data['final_impact_x_local'] for data in trajectory_data.values() if 'final_impact_x_local' in data]
all_final_y_local = [data['final_impact_y_local'] for data in trajectory_data.values() if 'final_impact_y_local' in data]

if all_final_x_local and all_final_y_local:
    mcmc_target_center_x = np.mean(all_final_x_local)
    mcmc_target_center_y = np.mean(all_final_y_local)
else:
    # Default to center of Nepal map if no rockets were simulated or trajectory data missing
    mcmc_target_center_x = NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2
    mcmc_target_center_y = NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2

# Parameters for the MCMC target distribution (2D Gaussian)
# Let's assume a standard deviation of 2 km (2000 meters) for impact precision.
target_mu = np.array([mcmc_target_center_x, mcmc_target_center_y])
target_cov = np.diag([2000**2, 2000**2]) # Covariance matrix for 2km std dev in x and y

# Define the target probability density function (PDF)
target_pdf = multivariate_normal(mean=target_mu, cov=target_cov)

# MCMC Parameters
num_samples = 20000
burn_in = 5000 # Discard initial samples
proposal_std = 500 # Standard deviation for the proposal distribution (in meters)

# MCMC Simulation (Metropolis-Hastings)
samples = []
current_sample = target_mu # Start at the mean of the target distribution

for i in range(num_samples + burn_in):
    # Propose a new sample
    proposal = current_sample + np.random.normal(0, proposal_std, size=2)

    # Calculate acceptance ratio
    prob_current = target_pdf.pdf(current_sample)
    prob_proposal = target_pdf.pdf(proposal)

    if prob_current == 0 and prob_proposal == 0: # Avoid division by zero if both are zero
        alpha = 1 # Treat as always accepted if outside high probability region
    elif prob_current == 0: # Current is zero, proposal is non-zero, always accept (to move towards higher density)
        alpha = 1
    else:
        alpha = min(1, prob_proposal / prob_current)

    # Accept or reject the proposal
    if np.random.rand() < alpha:
        current_sample = proposal
    
    if i >= burn_in:
        samples.append(current_sample)

samples_array = np.array(samples)

# Calculate Failure Rate for a 50 km^2 Area
TARGET_AREA_KM2 = 50
TARGET_AREA_M2 = TARGET_AREA_KM2 * 1e6 # Convert to square meters
TARGET_RADIUS_M = np.sqrt(TARGET_AREA_M2 / np.pi) # Radius of a circle with 50 km^2 area

# Calculate distances from the target center for each sample
distances_from_target = np.linalg.norm(samples_array - target_mu, axis=1)

# Count samples outside the target radius
failures = np.sum(distances_from_target > TARGET_RADIUS_M)
failure_rate_percent = (failures / len(samples_array)) * 100

print(f"\n--- MCMC Simulation Results ---")
print(f"Total MCMC samples: {len(samples_array)}")
print(f"Target area: {TARGET_AREA_KM2} km^2 (approx. radius {TARGET_RADIUS_M/1000:.2f} km)")
print(f"Number of samples outside target area (failures): {failures}")
print(f"Calculated Failure Rate: {failure_rate_percent:.2f}%")

# --- End MCMC Simulation Section ---


# --- PyDeck Visualization ---
# Collect all trajectory data and collision points for PyDeck after the Matplotlib part
pydeck_trajectory_data = {}
pydeck_collision_data = {}

for rocket_name, params in missiles_to_simulate.items():
    if rocket_name in trajectory_data: # Check if trajectory was successfully calculated
        pydeck_trajectory_data[rocket_name] = trajectory_data[rocket_name]
        pydeck_collision_data[rocket_name] = trajectory_data[rocket_name]['collisions']

# Call PyDeck visualization
# Need to pass MCMC results to PyDeck as well
mcmc_results_for_pydeck = {
    'samples_array': samples_array,
    'target_mu': target_mu,
    'target_radius': TARGET_RADIUS_M,
    'failure_rate_percent': failure_rate_percent
}
visualize_with_pydeck(pydeck_trajectory_data, pydeck_collision_data, mcmc_results_for_pydeck, TARGET_RADIUS_M)


# --- Folium Visualization ---
# Collect all trajectory data and collision points for Folium
folium_trajectory_data = {}
for rocket_name, data in trajectory_data.items():
    if rocket_name in trajectory_data:
        folium_trajectory_data[rocket_name] = {
            'global_lats': data['global_lats'],
            'global_lons': data['global_lons'],
            'global_alts': data['global_alts'], # Altitude is not directly used by Folium PolyLine for 3D, but good to have
            'color': data['color'],
            'collisions': data['collisions'],
            'final_impact_global_lat': data['final_impact_global_lat'],
            'final_impact_global_lon': data['final_impact_global_lon'],
            'crater_radius': data['crater_radius']
        }

# Pass MCMC results to Folium visualization
visualize_with_folium(folium_trajectory_data, samples_array, target_mu, TARGET_RADIUS_M, failure_rate_percent)
