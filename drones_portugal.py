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
    Prints the trajectory and returns the x, y coordinates along with the equation of motion.
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
    if initial_angle_rad < 0 or initial_angle_rad > np.pi / 2:
        raise ValueError("Launch angle must be between 0 and 90 degrees.")
    
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
        print("t:", t, "x:", x, "y:", y, "vx:", vx, "vy:", vy, "ax:", ax, "ay:", ay)
        return [vx, vy, ax, ay]

    # Event function to stop integration when the rocket hits the ground
    # This function returns True when the rocket's altitude (y) is less than or equal to 0
    # and is moving downwards (vy < 0).
    # This ensures we stop the simulation when the rocket impacts the ground.
    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    y0_state = [0, 0, v0 * np.cos(initial_angle_rad), v0 * np.sin(initial_angle_rad)]

    t_eval_max = 7000
    max_step_val = 1.0

    sol = solve_ivp(ode_system, (0, t_eval_max), y0_state, events=hit_ground, dense_output=True, max_step=max_step_val)
    print(f"Rocket: {rocket_name_pm}, Initial Speed: {v0:.2f} m/s, Launch Angle: {launch_angle_deg} degrees")
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
    print(f"Rocket: {rocket_name_pm}, Max Altitude: {max_altitude:.2f} km, Achieved Range: {achieved_range_m / 1000:.2f} km")
    print(f"Equation of Motion: {equation_str}")
    print(f"Current Speed: {current_speed_m_s / 1000:.2f} km/s, Launch Angle: {launch_angle_deg} degrees")
    print(f"Coordinates: {list(zip(x_vals, y_vals))}")
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
        print(f"Iteration {iteration+1}: Speed: {current_speed_km_s:.2f} km/s, Achieved Range: {achieved_range_km:.2f} km")

        if best_achieved_range_km == -1 or abs(achieved_range_km - target_range_km) < abs(best_achieved_range_km - target_range_km):
            best_achieved_range_km = achieved_range_km
            best_speed_km_s = current_speed_km_s
            best_x_vals, best_y_vals = x_vals, y_vals
            print(f"Iteration {iteration+1}: Best Speed: {best_speed_km_s:.2f} km/s, Achieved Range: {best_achieved_range_km:.2f} km")

        error_km = achieved_range_km - target_range_km

        if abs(error_km) < tolerance_km:
            rocket['speed_km_s'] = current_speed_km_s
            print(f"Target range achieved: {target_range_km} km with speed {current_speed_km_s:.2f} km/s")
            return x_vals, y_vals, current_speed_km_s, achieved_range_km

        if achieved_range_km > 0:
            scale_factor = (target_range_km / achieved_range_km)**0.5 
            scale_factor = np.clip(scale_factor, 0.85, 1.15)
            current_speed_km_s *= scale_factor
            print(f"Iteration {iteration+1}: Adjusting speed by scale factor: {scale_factor:.2f}")
        else:
            current_speed_km_s *= 1.5
            print("Iteration {iteration+1}: No range achieved, increasing speed by 50%")

        current_speed_km_s = np.clip(current_speed_km_s, low_speed_km_s, high_speed_km_s)
        
        if iteration > 5 and current_speed_km_s == low_speed_km_s and error_km < 0:
             break
        if iteration > 5 and current_speed_km_s == high_speed_km_s and error_km > 0:
             break

    rocket['speed_km_s'] = best_speed_km_s
    print(f"Final Speed for {rocket_name_adj}: {best_speed_km_s:.2f} km/s, Achieved Range: {best_achieved_range_km:.2f} km")
    return best_x_vals, best_y_vals, best_speed_km_s, best_achieved_range_km


# =============================================================================
# 3. Atmospheric Layer and Nepal Map Definitions
# =============================================================================

# Nepal Latitude and Longitude Bounds (approximate)
NEPAL_LAT_MIN = 26.3
NEPAL_LAT_MAX = 30.5
NEPAL_LON_MIN = 80.0
NEPAL_LON_MAX = 88.2
print("Nepal Latitude Bounds:", NEPAL_LAT_MIN, "to", NEPAL_LAT_MAX)
# New global constants for Nepal center
NEPAL_LAT_CENTER = (NEPAL_LAT_MIN + NEPAL_LAT_MAX) / 2
NEPAL_LON_CENTER = (NEPAL_LON_MIN + NEPAL_LON_MAX) / 2
print("Nepal Center Coordinates:", NEPAL_LAT_CENTER, NEPAL_LON_CENTER)
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
    [80.0, 26.3], [88.0, 26.3], [88.0, 30.0], [80.0, 30.0], [80.0, 26.3] # Basic rectangle
]
# A slightly more refined, but still simplified, outline of Nepal
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
    print(f"Converting Lat/Lon/Alt to Cartesian: ({lat_deg}, {lon_deg}, {alt_m}) -> ({x}, {y}, {z})")
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
    print(f"Converting Local Meters to Global Lat/Lon/Alt: ({local_x_meters}, {local_y_meters}, {altitude_meters}) -> ({global_lat_deg}, {global_lon_deg}, {altitude_meters})")
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
    print(f"Converting Global Lat/Lon to Local Meters: ({lat_deg}, {lon_deg}) -> ({local_x_meters}, {local_y_meters})")
    return local_x_meters, local_y_meters


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
print("Plotted Earth sphere with radius:", R_EARTH)
# Plot Nepal region on Earth's surface using hardcoded boundary
nepal_surface_x, nepal_surface_y, nepal_surface_z = [], [], []
for lon, lat in NEPAL_BOUNDARY_COORDS_LON_LAT:
    x, y, z = convert_lat_lon_alt_to_cartesian(lat, lon, 0)
    nepal_surface_x.append(x)
    nepal_surface_y.append(y)
    nepal_surface_z.append(z)
ax.plot(nepal_surface_x, nepal_surface_y, nepal_surface_z, color='darkgreen', linewidth=2, label='Nepal Region')
print("Plotted Nepal boundary on Earth's surface.")

# Satellite position (animated to orbit slightly)
satellite_alt_m = 400e3 # 400 km
# Initial satellite position (fixed for now, will animate later)
sat_lat_init, sat_lon_init = NEPAL_LAT_CENTER + 2, NEPAL_LON_CENTER + 2 # Slightly offset from Nepal center
sat_x_init, sat_y_init, sat_z_init = convert_lat_lon_alt_to_cartesian(sat_lat_init, sat_lon_init, satellite_alt_m)
satellite_plot, = ax.plot([sat_x_init], [sat_y_init], [sat_z_init], 's', color='cyan', markersize=8, label='Satellite') # 's' for square marker
print("Plotted initial satellite position at altitude:", satellite_alt_m)
# Function to update satellite position (simple circular orbit)
def update_satellite_position(frame):
    # Simple circular orbit around Nepal center
    orbit_radius = R_EARTH + satellite_alt_m
    angle = frame * (2 * np.pi / 360)  # Complete one orbit in 360 frames
    sat_x = orbit_radius * np.cos(angle) * np.cos(np.radians(NEPAL_LAT_CENTER))
    sat_y = orbit_radius * np.sin(angle) * np.cos(np.radians(NEPAL_LAT_CENTER))
    sat_z = orbit_radius * np.sin(np.radians(NEPAL_LAT_CENTER))
    satellite_plot.set_data_3d([sat_x], [sat_y], [sat_z])
    return satellite_plot,

# Initialize the satellite position
update_satellite_position(0)  # Set initial position
# Plot Nepal boundary as a 3D polygon on the Earth's surface
print("Plotting Nepal boundary as a 3D polygon on Earth's surface.")
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
nepal_boundary_poly = Poly3DCollection([list(zip(nepal_surface_x, nepal_surface_y, nepal_surface_z))], color='darkgreen', alpha=0.5)
ax.add_collection3d(nepal_boundary_poly)
# Set the aspect ratio to be equal for a spherical Earth
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for a sphere
# Set plot limits to encompass the Earth sphere
ax.set_xlim(-R_EARTH * 1.2, R_EARTH * 1.2)
ax.set_ylim(-R_EARTH * 1.2, R_EARTH * 1.2)
ax.set_zlim(-R_EARTH * 1.2, R_EARTH * 1.2)
# Set labels and title
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Rocket Trajectories and Atmospheric Layers over Earth (Matplotlib)')
ax.legend()
# Animation function to update the satellite position
def animate(frame):
    # Update satellite position
    update_satellite_position(frame)
    
    # Update trajectories for each rocket
    for rocket_name in missiles_to_simulate:
        if rocket_name in trajectory_lines:
            x_global = trajectory_data[rocket_name]['x_global']
            y_global = trajectory_data[rocket_name]['y_global']
            z_global = trajectory_data[rocket_name]['z_global']
            trajectory_lines[rocket_name].set_data_3d(x_global[:frame], y_global[:frame], z_global[:frame])
            trajectory_points[rocket_name].set_data_3d(x_global[frame-1:frame], y_global[frame-1:frame], z_global[frame-1:frame])
            
            # Update collision markers
            collision_points = trajectory_data[rocket_name]['collisions']
            if frame < len(collision_points):
                coll_x, coll_y, coll_z, _ = collision_points[frame]
                collision_markers[rocket_name].set_data_3d([coll_x], [coll_y], [coll_z])
            else:
                collision_markers[rocket_name].set_data_3d([], [], [])
    print(f"Animating frame {frame}: Updated satellite and rocket trajectories.")
    # Create animation
    ani = FuncAnimation(fig, animate, frames=NUM_ANIMATION_FRAMES, interval=ANIMATION_INTERVAL_MS, blit=False)
    # Initialize trajectory lines and points
    trajectory_lines = {}

    return satellite_plot, *trajectory_lines.values(), *trajectory_points.values(), *collision_markers.values()
# =============================================================================
# 5. Plot Atmospheric Layers (as spherical segments over Nepal)

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

    # Note: x_traj is range, y_traj is altitude.
    # For local 2D map, we use x_traj as x and a constant y for the map.
    # The MCMC section will use these local coordinates.
    local_map_x_for_mcmc = x_traj + convert_global_lat_lon_to_local_meters(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, NEPAL_LAT_MIN, NEPAL_LON_MIN)[0]
    local_map_y_for_mcmc = np.full_like(x_traj, convert_global_lat_lon_to_local_meters(NEPAL_LAT_CENTER, NEPAL_LON_CENTER, NEPAL_LAT_MIN, NEPAL_LON_MIN)[1])
    
    # Convert local trajectory points to global lat/lon/alt
    global_lats, global_lons, global_alts = [], [], []
    for i in range(len(x_traj)):
        # x_traj[i] is the horizontal distance from launch, local_y_meters is 0 for a straight line launch
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
        'x_local_mcmc': local_map_x_for_mcmc, # Local x for MCMC target
        'y_local_mcmc': local_map_y_for_mcmc, # Local y for MCMC target
        'z_local_altitude': y_traj, # Altitude for local context
        'color': params["color"],
        'length': len(x_traj),
        'final_impact_x_local': local_map_x_for_mcmc[-1], # Use local for MCMC target
        'final_impact_y_local': local_map_y_for_mcmc[-1] # Use local for MCMC target
    }
    print(f"Calculated trajectory for {rocket_name}:")
    print(f"  Final Speed: {final_speed:.2f} km/s, Achieved Range: {achieved_range:.2f} km")
    print(f"  Global Coordinates: {list(zip(global_X, global_Y, global_Z))[:50]}... (showing first 5 points)")
    print(f"  Local MCMC Coordinates: {list(zip(local_map_x_for_mcmc, local_map_y_for_mcmc[:50]))}... (showing first 5 points)")
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
                coll_local_x_range = x_traj[idx] + fraction * (x_traj[idx+1] - x_traj[idx])
                coll_local_y_alt = min_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(coll_local_x_range, 0, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Entry {layer_name}'))
                print(f"Collision Entry {layer_name} for {rocket_name}: ({coll_global_X}, {coll_global_Y}, {coll_global_Z})")
                # Add marker for entry point
                collision_markers[rocket_name] = ax.plot([], [], [], 'o', color=params["color"], markersize=5, label=f'{rocket_name} Collisions')[0]
                print(f"The coordinates of the entry point for {rocket_name} in layer {layer_name} are: ({coll_global_X}, {coll_global_Y}, {coll_global_Z})")
        
        # Check for exit from layer (from below max_alt to above max_alt)
        exit_indices = np.where((y_traj[:-1] < max_alt) & (y_traj[1:] >= max_alt))[0]
        for idx in exit_indices:
            if y_traj[idx+1] - y_traj[idx] != 0:
                fraction = (max_alt - y_traj[idx]) / (y_traj[idx+1] - y_traj[idx])
                coll_local_x_range = x_traj[idx] + fraction * (x_traj[idx+1] - x_traj[idx])
                coll_local_y_alt = max_alt # This is the altitude of collision

                # Convert to global lat/lon/alt
                coll_global_lat, coll_global_lon, coll_global_alt = convert_local_meters_to_global_lat_lon_alt(coll_local_x_range, 0, coll_local_y_alt, launch_lat_deg, launch_lon_deg)
                # Convert to global XYZ
                coll_global_X, coll_global_Y, coll_global_Z = convert_lat_lon_alt_to_cartesian(coll_global_lat, coll_global_lon, coll_global_alt)
                collision_points_for_rocket.append((coll_global_X, coll_global_Y, coll_global_Z, f'Exit {layer_name}'))
                print(f"Collision Exit {layer_name} for {rocket_name}: ({coll_global_X}, {coll_global_Y}, {coll_global_Z})")
                print(f"The coordinates of the exit point for {rocket_name} in layer {layer_name} are: ({coll_global_X}, {coll_global_Y}, {coll_global_Z})")

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
    satellite_plot.set_data_3d([], [], []) # Initialize satellite
    print("Initialized animation with empty trajectories and satellite position.")
    # Return all artists for the initial frame
    return list(trajectory_lines.values()) + list(trajectory_points.values()) + list(collision_markers.values()) + [satellite_plot]

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
        print(f"Updating trajectory for {rocket_name} at frame {frame}: Current Index: {current_idx}")
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
        print(f"Updated collision markers for {rocket_name} at frame {frame}: {len(coll_x_display)} points displayed.")

    # Animate satellite position (simple orbit around Earth)
    # This is a simplified animation, just rotating around the Z-axis
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
# Use the maximum length of any trajectory to ensure all are animated fully
max_traj_length = max(data['length'] for data in trajectory_data.values()) if trajectory_data else 100
total_frames = max_traj_length # Animate until the longest trajectory is complete

anim = FuncAnimation(fig, update_animation, frames=total_frames,
                     init_func=init_animation, blit=False, interval=ANIMATION_INTERVAL_MS, repeat=False)
print("Animation initialized with total frames:", total_frames)
# Show the plot
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
    print(f"Converting Meters to Lat/Lon: ({x_meters}, {y_meters}) -> ({lat}, {lon})")
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
            print(f"Converting Cartesian to Lat/Lon/Alt: ({x}, {y}, {z}) -> ({np.degrees(lat_rad)}, {np.degrees(lon_rad)}, {alt_m})")
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

    # 4. Collision Markers
    all_collision_points = []
    for rocket_name, collisions in collision_markers_data_pydeck.items():
        for coll_global_x, coll_global_y, coll_global_z, coll_type in collisions:
            # These are already global Cartesian. Need to convert back to lat/lon/alt for PyDeck.
            def convert_cartesian_to_lat_lon_alt(x, y, z):
                r = np.sqrt(x**2 + y**2 + z**2)
                lat_rad = np.arcsin(z / r)
                lon_rad = np.arctan2(y, x)
                alt_m = r - R_EARTH
                print(f"Converting Cartesian to Lat/Lon/Alt: ({x}, {y}, {z}) -> ({np.degrees(lat_rad)}, {np.degrees(lon_rad)}, {alt_m})")
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
    print(f"Added {len(all_collision_points)} collision markers to PyDeck layers.")
    # Add Satellite to PyDeck (as a single point)
    sat_lat_pydeck, sat_lon_pydeck, sat_alt_pydeck = convert_local_meters_to_global_lat_lon_alt(0, 0, satellite_alt_m, sat_lat_init, sat_lon_init)
    satellite_pydeck_df = pd.DataFrame([{
        'position': [sat_lon_pydeck, sat_lat_pydeck, sat_alt_pydeck],
        'color': [0, 255, 255, 255], # Cyan
        'size': 5000, # Larger size for visibility
        'name': 'Satellite'
    }])
    print(f"Satellite position in PyDeck: ({sat_lon_pydeck}, {sat_lat_pydeck}, {sat_alt_pydeck})")
    # Add satellite position to PyDeck layers
    print("Adding satellite position to PyDeck layers...")
    # Create a ScatterplotLayer for the satellite position
    # This uses the same conversion function to get lat/lon/alt for PyDeck
    satellite_pydeck_df['position'] = satellite_pydeck_df['position'].apply(lambda pos: [pos[0], pos[1], pos[2]]) # Ensure correct format
    satellite_pydeck_df['color'] = satellite_pydeck_df['color'].apply(lambda c: [int(c[0]), int(c[1]), int(c[2]), int(c[3])]) # Ensure RGBA format
    satellite_pydeck_df['size'] = satellite_pydeck_df['size'].apply(lambda s: int(s)) # Ensure size is an integer
    # Add the satellite layer to the layers list
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
    print("Added satellite position to PyDeck layers.")


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
        print(f"Prepared trajectory data for {rocket_name} for PyDeck visualization.")
    else:
        print(f"Warning: {rocket_name} trajectory data not found for PyDeck visualization.")

# Call PyDeck visualization
visualize_with_pydeck(pydeck_trajectory_data, pydeck_collision_data)


# =============================================================================
# 6. MCMC Simulation for Impact Probability and Failure Rate
# =============================================================================

# Calculate the average final impact point from the simulated rockets (using local coordinates)
all_final_x_local = [data['final_impact_x_local'] for data in trajectory_data.values() if 'final_impact_x_local' in data]
all_final_y_local = [data['final_impact_y_local'] for data in trajectory_data.values() if 'final_impact_y_local' in data]
print(f"Final impact points (local coordinates): {list(zip(all_final_x_local, all_final_y_local))}")
if all_final_x_local and all_final_y_local:
    mcmc_target_center_x = np.mean(all_final_x_local)
    mcmc_target_center_y = np.mean(all_final_y_local)
else:
    # Default to center of Nepal map if no rockets were simulated or trajectory data missing
    mcmc_target_center_x = NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2
    mcmc_target_center_y = NEPAL_Y_MIN + (NEPAL_Y_MAX - NEPAL_Y_MIN) / 2
print(f"Calculated MCMC target center (local coordinates): ({mcmc_target_center_x}, {mcmc_target_center_y})")
# Parameters for the MCMC target distribution (2D Gaussian)
# Let's assume a standard deviation of 2 km (2000 meters) for impact precision.
target_mu = np.array([mcmc_target_center_x, mcmc_target_center_y])
target_cov = np.diag([2000**2, 2000**2]) # Covariance matrix for 2km std dev in x and y
print(f"Target mean (mu): {target_mu}, Target covariance (cov): {target_cov}")
# Define the target probability density function (PDF)
target_pdf = multivariate_normal(mean=target_mu, cov=target_cov)
print(f"Target PDF initialized with mean {target_mu} and covariance {target_cov}")
# MCMC Parameters
num_samples = 20000
burn_in = 5000 # Discard initial samples
proposal_std = 500 # Standard deviation for the proposal distribution (in meters)

# MCMC Simulation (Metropolis-Hastings)
samples = []
current_sample = target_mu # Start at the mean of the target distribution
print(f"Starting MCMC simulation with {num_samples} samples and burn-in of {burn_in}...")
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
    print(f"Iteration {i+1}/{num_samples + burn_in}: Current Sample: {current_sample}, Proposal: {proposal}, Acceptance Ratio: {alpha:.4f}")
    # Print progress every 1000 iterations
    if (i + 1) % 1000 == 0:
        print(f"Progress: {i + 1}/{num_samples + burn_in} samples processed.")
    # Print the current sample and proposal for debugging
    print(f"Current Sample: {current_sample}, Proposal: {proposal}, Acceptance Ratio: {alpha:.4f}")
    # Print the acceptance ratio for debugging
    # Print the proposal and current sample for debugging
    print(f"Proposal: {proposal}, Current Sample: {current_sample}, Acceptance Ratio: {alpha:.4f}")

    # Accept or reject the proposal
    if np.random.rand() < alpha:
        current_sample = proposal
    
    if i >= burn_in:
        samples.append(current_sample)

samples_array = np.array(samples)
print(f"MCMC simulation completed. Total accepted samples: {len(samples_array)}")
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
print(f"Creating MCMC plot with {len(samples_array)} samples...")
# Plot Nepal ground plane (2D representation for MCMC) using hardcoded boundary
nepal_boundary_local_x, nepal_boundary_local_y = [], []
# Assuming the MCMC plot origin is NEPAL_LON_MIN, NEPAL_LAT_MIN
mcmc_origin_lat = NEPAL_LAT_MIN
mcmc_origin_lon = NEPAL_LON_MIN
print(f"Converting Nepal boundary coordinates to local meters with origin ({mcmc_origin_lat}, {mcmc_origin_lon})...")
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
