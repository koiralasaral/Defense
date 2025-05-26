import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math # For math.radians
import pydeck as pdk # Import pydeck
import pandas as pd # Import pandas for DataFrame
import matplotlib.colors # For converting named colors to RGBA
from scipy.integrate import solve_ivp # Required for projectile motion
from scipy.stats import multivariate_normal # For MCMC target distribution

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

    mask = y_vals >= -1
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
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

    return x_vals, y_vals, equation_str, max_altitude, launch_angle_deg, achieved_range_m

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

    low_speed_km_s = 0.1
    high_speed_km_s = 30

    for iteration in range(max_iterations):
        current_speed_m_s = current_speed_km_s * 1000
        
        x_vals, y_vals, _, _, _, achieved_range_m = projectile_motion(
            rocket_name_adj, current_speed_m_s, initial_launch_angle_deg, atmosphere_model='spherical'
        )
        achieved_range_km = achieved_range_m / 1000

        if best_achieved_range_km == -1 or abs(achieved_range_km - target_range_km) < abs(best_achieved_range_km - target_range_km):
            best_achieved_range_km = achieved_range_km
            best_speed_km_s = current_speed_km_s
            best_x_vals, best_y_vals = x_vals, y_vals

        error_km = achieved_range_km - target_range_km

        if abs(error_km) < tolerance_km:
            rocket['speed_km_s'] = current_speed_km_s
            return x_vals, y_vals, current_speed_km_s, achieved_range_km

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
    return best_x_vals, best_y_vals, best_speed_km_s, best_achieved_range_km


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

# Plot Nepal region on Earth's surface
# Convert Nepal's corners to global XYZ for plotting
nepal_corners_lat_lon = [
    (NEPAL_LAT_MIN, NEPAL_LON_MIN),
    (NEPAL_LAT_MIN, NEPAL_LON_MAX),
    (NEPAL_LAT_MAX, NEPAL_LON_MAX),
    (NEPAL_LAT_MAX, NEPAL_LON_MIN),
    (NEPAL_LAT_MIN, NEPAL_LON_MIN) # Close the polygon
]
nepal_surface_x, nepal_surface_y, nepal_surface_z = [], [], []
for lat, lon in nepal_corners_lat_lon:
    x, y, z = convert_lat_lon_alt_to_cartesian(lat, lon, 0)
    nepal_surface_x.append(x)
    nepal_surface_y.append(y)
    nepal_surface_z.append(z)
ax.plot(nepal_surface_x, nepal_surface_y, nepal_surface_z, color='darkgreen', linewidth=2, label='Nepal Region')


# Satellite position
satellite_alt_m = 400e3 # 400 km
sat_lat, sat_lon = NEPAL_LAT_CENTER + 2, NEPAL_LON_CENTER + 2 # Slightly offset from Nepal center
sat_x, sat_y, sat_z = convert_lat_lon_alt_to_cartesian(sat_lat, sat_lon, satellite_alt_m)
ax.plot([sat_x], [sat_y], [sat_z], 's', color='cyan', markersize=8, label='Satellite') # 's' for square marker


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
    "V2": {"color": 'red', "launch_angle": 60}, # Higher angle to emphasize altitude
    "Iskander": {"color": 'orange', "launch_angle": 55},
    "Minuteman_III": {"color": 'blue', "launch_angle": 45},
    "DF_41": {"color": 'purple', "launch_angle": 40},
    "Sarmat": {"color": 'black', "launch_angle": 35},
}

trajectory_data = {}
trajectory_lines = {}
trajectory_points = {}
collision_markers = {}

# Pre-calculate trajectories and collision points
for rocket_name, params in missiles_to_simulate.items():
    # Ensure rocket_data has the adjusted speed from the previous run
    if rocket_name not in rocket_data:
        print(f"Warning: {rocket_name} not found in rocket_data. Skipping trajectory calculation.")
        continue

    # Adjust speed if not already done (this will use the target range from global config)
    # This ensures the rocket_data has the latest adjusted speed
    target_range = TARGET_RANGES_KM.get(rocket_name, rocket_data[rocket_name]['range_km'])
    x_traj, y_traj, final_speed, achieved_range = adjust_speed_for_range(
        rocket_name, target_range, initial_launch_angle_deg=params["launch_angle"]
    )
    
    # For visualization, we will launch from the center of Nepal for global projection
    launch_lat_deg = NEPAL_LAT_CENTER
    launch_lon_deg = NEPAL_LON_CENTER

    # Adjust trajectory coordinates to be relative to the map for local MCMC
    # Note: x_traj is range, y_traj is altitude.
    # For local 2D map, we can use x_traj as x and a constant y for the map.
    local_map_x = x_traj + (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2) # Offset to center of Nepal's x-range
    local_map_y = np.full_like(x_traj, NEPAL_Y_MIN) # Keep y-coord constant for 2D trajectory on map

    # Convert local trajectory points to global lat/lon/alt
    global_lats, global_lons, global_alts = [], [], []
    for i in range(len(x_traj)):
        # x_traj[i] is the horizontal distance from launch, local_map_y[i] is effectively 0 for horizontal spread
        lat, lon, alt = convert_local_meters_to_global_lat_lon_alt(x_traj[i], 0, y_traj[i], launch_lat_deg, launch_lon_deg)
        global_lats.append(lat)
        global_lons.append(lon)
        global_alts.append(alt)

    # Convert global lat/lon/alt to global Cartesian (X,Y,Z) for plotting on sphere
    global_X, global_Y, global_Z = [], [], []
    for i in range(len(global_lats)):
        x, y, z = convert_lat_lon_alt_to_cartesian(global_lats[i], global_lons[i], global_alts[i])
        global_X.append(x)
        global_Y.append(y)
        global_Z.append(z)

    trajectory_data[rocket_name] = {
        'x_global': np.array(global_X),
        'y_global': np.array(global_Y),
        'z_global': np.array(global_Z),
        'x_local': local_map_x, # Local x for MCMC target
        'y_local': local_map_y, # Local y for MCMC target
        'z_local': y_traj, # Altitude for local context
        'color': params["color"],
        'length': len(x_traj),
        'final_impact_x_local': local_map_x[-1], # Use local for MCMC target
        'final_impact_y_local': local_map_y[-1] # Use local for MCMC target
    }
    
    # Initialize plots for animation (use global coordinates)
    line, = ax.plot([], [], [], '-', color=params["color"], linewidth=2, label=f'{rocket_name} Trajectory')
    point, = ax.plot([], [], [], 'o', color=params["color"], markersize=5)
    trajectory_lines[rocket_name] = line
    trajectory_points[rocket_name] = point

    # Find collision points with atmospheric layers (need to convert these to global XYZ too)
    collision_points_for_rocket = []
    for layer_name, layer_params in ATMOSPHERE_LAYERS.items():
        min_alt = layer_params["min_alt"]
        max_alt = layer_params["max_alt"]

        # Check for entry into layer (from below min_alt to above min_alt)
        entry_indices = np.where((y_traj[:-1] < min_alt) & (y_traj[1:] >= min_alt))[0]
        for idx in entry_indices:
            # Interpolate to find precise intersection point in local coords
            if y_traj[idx+1] - y_traj[idx] != 0:
                fraction = (min_alt - y_traj[idx]) / (y_traj[idx+1] - y_traj[idx])
                coll_local_x = x_traj[idx] + fraction * (x_traj[idx+1] - x_traj[idx])
                coll_local_y_alt = min_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(coll_local_x, 0, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Entry {layer_name}'))
        
        # Check for exit from layer (from below max_alt to above max_alt)
        exit_indices = np.where((y_traj[:-1] < max_alt) & (y_traj[1:] >= max_alt))[0]
        for idx in exit_indices:
            if y_traj[idx+1] - y_traj[idx] != 0:
                fraction = (max_alt - y_traj[idx]) / (y_traj[idx+1] - y_traj[idx])
                coll_local_x = x_traj[idx] + fraction * (x_traj[idx+1] - x_traj[idx])
                coll_local_y_alt = max_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(coll_local_x, 0, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Exit {layer_name}'))

    collision_markers[rocket_name] = ax.plot([], [], [], 'X', color=params["color"], markersize=8, label=f'{rocket_name} Collisions')[0]
    trajectory_data[rocket_name]['collisions'] = collision_points_for_rocket


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
    return list(trajectory_lines.values()) + list(trajectory_points.values()) + list(collision_markers.values())

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
        for coll_global_x, coll_global_y, coll_global_z, _ in data['collisions']:
            # A simplified check for showing collision markers based on current rocket position
            # This assumes a general forward motion.
            current_rocket_pos = np.array([data['x_global'][current_idx], data['y_global'][current_idx], data['z_global'][current_idx]])
            collision_point_pos = np.array([coll_global_x, coll_global_y, coll_global_z])
            
            # Check if the rocket has passed the collision point in terms of distance from launch
            # This is a heuristic and might not be perfect for complex trajectories
            if np.linalg.norm(current_rocket_pos - convert_lat_lon_alt_to_cartesian(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, 0)) > \
               np.linalg.norm(collision_point_pos - convert_lat_lon_alt_to_cartesian(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, 0)):
                coll_x_display.append(coll_global_x)
                coll_y_display.append(coll_global_y)
                coll_z_display.append(coll_global_z)
        collision_markers[rocket_name].set_data_3d(coll_x_display, coll_y_display, coll_z_display)
        artists.append(collision_markers[rocket_name])

    # Rotate camera for better view
    ax.view_init(elev=20, azim=frame * 0.5) # Slower rotation

    return artists

# Determine total frames for animation
# Use the maximum length of any trajectory to ensure all are animated fully
max_traj_length = max(data['length'] for data in trajectory_data.values()) if trajectory_data else 100
total_frames = max_traj_length # Animate until the longest trajectory is complete

anim = FuncAnimation(fig, update_animation, frames=total_frames,
                     init_func=init_animation, blit=False, interval=ANIMATION_INTERVAL_MS, repeat=False)

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

def visualize_with_pydeck(trajectory_data_pydeck, collision_markers_data_pydeck):
    # Base map view centered over Nepal
    view_state = pdk.ViewState(
        latitude=(NEPAL_LAT_MIN + NEPAL_LAT_MAX) / 2,
        longitude=(NEPAL_LON_MIN + NEPAL_LON_MAX) / 2,
        zoom=5,
        pitch=45,
        bearing=0
    )

    layers = []

    # 1. Nepal Ground Plane (as a single polygon)
    # Define the corners of Nepal's bounding box in lat/lon
    nepal_polygon_coords = [
        [NEPAL_LON_MIN, NEPAL_LAT_MIN],
        [NEPAL_LON_MAX, NEPAL_LAT_MIN],
        [NEPAL_LON_MAX, NEPAL_LAT_MAX],
        [NEPAL_LON_MIN, NEPAL_LAT_MAX],
        [NEPAL_LON_MIN, NEPAL_LAT_MIN] # Close the polygon
    ]
    ground_df = pd.DataFrame({
        'polygon': [nepal_polygon_coords],
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
        for i in range(len(data['x_global'])):
            # PyDeck expects [longitude, latitude, altitude]
            # Convert global Cartesian back to lat/lon for PyDeck if needed, or if already in lat/lon, use that.
            # Here, data['x_global'], data['y_global'], data['z_global'] are Cartesian relative to Earth's center.
            # PyDeck needs lat/lon/alt. We need to re-convert from global XYZ to lat/lon/alt.
            # This is a bit circular, but necessary if the source data for PyDeck is the global Cartesian.
            # Let's use the stored global_lats, global_lons, global_alts from the Matplotlib calculation.
            # Assuming these are available in trajectory_data_pydeck.
            
            # Re-calculating lat/lon from global XYZ for PyDeck just to be safe, though less efficient
            # This requires a function to convert Cartesian back to Spherical.
            # For simplicity, let's assume the `global_lats`, `global_lons`, `global_alts` are directly stored
            # in `trajectory_data` when it's populated for Matplotlib.
            # For this example, we'll assume `trajectory_data_pydeck` has `global_lats`, `global_lons`, `global_alts`
            # which were calculated when `trajectory_data` was initially filled.
            
            # Let's modify trajectory_data to store global_lats, global_lons, global_alts directly.
            # This was already done in the previous step. So we can use them directly.
            lat = data.get('global_lats_original', 0) # Placeholder, assuming they are stored
            lon = data.get('global_lons_original', 0) # Placeholder
            alt = data.get('global_alts_original', 0) # Placeholder

            # Correction: The `trajectory_data` already stores `x_global`, `y_global`, `z_global` as Cartesian.
            # PyDeck's PathLayer expects `[longitude, latitude, altitude]`.
            # We need to convert the original local `x_traj`, `y_traj` (altitude) to `lat/lon/alt` for PyDeck.
            # This is what `convert_local_meters_to_global_lat_lon_alt` does.
            # Let's use the original `x_traj` (range) and `y_traj` (altitude) directly.
            
            # Re-calculate lat/lon/alt for PyDeck using the local trajectory data
            # The `x_traj` (range) and `y_traj` (altitude) are the core.
            # We need the launch point for this conversion.
            launch_lat_deg = NEPAL_LAT_CENTER
            launch_lon_deg = NEPAL_LON_CENTER
            
            # The `trajectory_data` dict needs to store the original `x_traj` and `y_traj` too.
            # Let's assume `trajectory_data[rocket_name]['original_x_traj']` and `['original_y_traj']` exist.
            # Or, more simply, use the `x_local` (range) and `z_local` (altitude) from `trajectory_data`.
            
            # Get the local range and altitude from the trajectory_data
            local_range = data['x_local'][i] - (NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2) # Remove launch offset
            local_altitude = data['z_local'][i]

            lat, lon, alt = convert_local_meters_to_global_lat_lon_alt(local_range, 0, local_altitude, launch_lat_deg, launch_lon_deg)
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

    # 4. Collision Markers
    all_collision_points = []
    for rocket_name, collisions in collision_markers_data_pydeck.items():
        for coll_global_x, coll_global_y, coll_global_z, coll_type in collisions:
            # These are already global Cartesian. Need to convert back to lat/lon/alt for PyDeck.
            # This requires a function to convert Cartesian to Spherical.
            # For simplicity, let's re-use the `convert_local_meters_to_global_lat_lon_alt` logic
            # by noting that `coll_global_x`, `coll_global_y`, `coll_global_z` are from
            # `convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)`.
            # So we need to reconstruct `coll_global_lat`, `coll_global_lon`, `coll_global_alt`.
            # This is getting complicated. Let's simplify the collision data structure for PyDeck.
            
            # The collision points in `trajectory_data[rocket_name]['collisions']` are already in
            # `(coll_global_X, coll_global_Y, coll_global_Z)` format.
            # We need to convert them back to `(lon, lat, alt)` for PyDeck.
            # Let's add a helper function for Cartesian to Spherical.

            def convert_cartesian_to_lat_lon_alt(x, y, z):
                r = np.sqrt(x**2 + y**2 + z**2)
                lat_rad = np.arcsin(z / r)
                lon_rad = np.arctan2(y, x)
                alt_m = r - R_EARTH
                return np.degrees(lat_rad), np.degrees(lon_rad), alt_m

            lat, lon, alt = convert_cartesian_to_lat_lon_alt(coll_global_x, coll_global_y, coll_global_z)
            
            all_collision_points.append({
                'position': [lon, lat, alt],
                'color': [255, 0, 0, 255], # Red 'X'
                'size': 100, # Size in meters
                'name': f"{rocket_name} {coll_type}"
            })
    
    if all_collision_points:
        collision_df = pd.DataFrame(all_collision_points)
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


# Main execution flow (kept as is, but now calls PyDeck at the end)

# Pre-calculate trajectories and collision points for Matplotlib and PyDeck
# This block already exists in the original script and populates `trajectory_data`
# and `collision_markers` (which is implicitly used via `trajectory_data[rocket_name]['collisions']`).
# No changes needed here, just ensuring it runs before PyDeck visualization.

# --- Matplotlib Animation ---
# (The existing Matplotlib animation code block goes here)
# ...

# --- PyDeck Visualization ---
# Collect all trajectory data and collision points for PyDeck after the Matplotlib part
pydeck_trajectory_data = {}
pydeck_collision_data = {}

for rocket_name, params in missiles_to_simulate.items():
    if rocket_name in trajectory_data: # Check if trajectory was successfully calculated
        pydeck_trajectory_data[rocket_name] = trajectory_data[rocket_name]
        pydeck_collision_data[rocket_name] = trajectory_data[rocket_name]['collisions']

# Call PyDeck visualization
visualize_with_pydeck(pydeck_trajectory_data, pydeck_collision_data)


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

# Plot Nepal ground plane (2D representation for MCMC)
# We need to create a simple 2D rectangle for Nepal's bounds for this plot.
nepal_rect_x = [NEPAL_X_MIN, NEPAL_X_MAX, NEPAL_X_MAX, NEPAL_X_MIN, NEPAL_X_MIN]
nepal_rect_y = [NEPAL_Y_MIN, NEPAL_Y_MIN, NEPAL_Y_MAX, NEPAL_Y_MAX, NEPAL_Y_MIN]
ax_mcmc.plot(nepal_rect_x, nepal_rect_y, color='green', linewidth=2, label='Nepal Region (2D)')
ax_mcmc.fill(nepal_rect_x, nepal_rect_y, color='green', alpha=0.2)


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
