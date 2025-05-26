import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import math # For math.radians
from scipy.integrate import solve_ivp
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

# Conversion factors (approximate)
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON_AT_NEPAL_AVG_LAT = 111.0 * math.cos(math.radians((NEPAL_LAT_MIN + NEPAL_LAT_MAX) / 2))

# Convert lat/lon bounds to meters for plotting
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

# =============================================================================
# 4. Main Animation Logic
# =============================================================================

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Set initial camera view
ax.view_init(elev=20, azim=20)

# Plot Nepal ground plane
map_x = np.linspace(NEPAL_X_MIN, NEPAL_X_MAX, 2)
map_y = np.linspace(NEPAL_Y_MIN, NEPAL_Y_MAX, 2)
X_map, Y_map = np.meshgrid(map_x, map_y)
Z_map = np.zeros_like(X_map)
ax.plot_surface(X_map, Y_map, Z_map, color='green', alpha=0.7, label='Nepal Ground')

# Plot Atmospheric Layers
layer_surfaces = {}
for layer_name, params in ATMOSPHERE_LAYERS.items():
    min_alt = params["min_alt"]
    max_alt = params["max_alt"]
    color = params["color"]
    alpha = params["alpha"]

    # Bottom surface of the layer
    Z_bottom = np.full_like(X_map, min_alt)
    layer_surfaces[f"{layer_name}_bottom"] = ax.plot_surface(X_map, Y_map, Z_bottom, color=color, alpha=alpha)
    
    # Top surface of the layer
    Z_top = np.full_like(X_map, max_alt)
    layer_surfaces[f"{layer_name}_top"] = ax.plot_surface(X_map, Y_map, Z_top, color=color, alpha=alpha)

    # Side walls (simplified for rectangular region)
    # Front wall
    ax.plot([NEPAL_X_MIN, NEPAL_X_MAX], [NEPAL_Y_MIN, NEPAL_Y_MIN], [min_alt, min_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MIN, NEPAL_X_MAX], [NEPAL_Y_MIN, NEPAL_Y_MIN], [max_alt, max_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MIN, NEPAL_X_MIN], [NEPAL_Y_MIN, NEPAL_Y_MIN], [min_alt, max_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MAX, NEPAL_X_MAX], [NEPAL_Y_MIN, NEPAL_Y_MIN], [min_alt, max_alt], color=color, alpha=alpha)
    # Back wall
    ax.plot([NEPAL_X_MIN, NEPAL_X_MAX], [NEPAL_Y_MAX, NEPAL_Y_MAX], [min_alt, min_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MIN, NEPAL_X_MAX], [NEPAL_Y_MAX, NEPAL_Y_MAX], [max_alt, max_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MIN, NEPAL_X_MIN], [NEPAL_Y_MAX, NEPAL_Y_MAX], [min_alt, max_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MAX, NEPAL_X_MAX], [NEPAL_Y_MAX, NEPAL_Y_MAX], [min_alt, max_alt], color=color, alpha=alpha)
    # Left wall
    ax.plot([NEPAL_X_MIN, NEPAL_X_MIN], [NEPAL_Y_MIN, NEPAL_Y_MAX], [min_alt, min_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MIN, NEPAL_X_MIN], [NEPAL_Y_MIN, NEPAL_Y_MAX], [max_alt, max_alt], color=color, alpha=alpha)
    # Right wall
    ax.plot([NEPAL_X_MAX, NEPAL_X_MAX], [NEPAL_Y_MIN, NEPAL_Y_MAX], [min_alt, min_alt], color=color, alpha=alpha)
    ax.plot([NEPAL_X_MAX, NEPAL_X_MAX], [NEPAL_Y_MIN, NEPAL_Y_MAX], [max_alt, max_alt], color=color, alpha=alpha)


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
    
    # For visualization, we will launch from a point within Nepal's bounds
    # For simplicity, launch from the center-bottom of the map.
    launch_x_offset = NEPAL_X_MIN + (NEPAL_X_MAX - NEPAL_X_MIN) / 2
    launch_y_offset = NEPAL_Y_MIN # Launch from the southern border

    # Adjust trajectory coordinates to be relative to the map
    x_traj_shifted = x_traj + launch_x_offset
    y_traj_shifted = np.full_like(x_traj, launch_y_offset) # Keep y-coord constant for 2D trajectory on map
    z_traj = y_traj # The y-component from projectile_motion is altitude (z-axis)

    trajectory_data[rocket_name] = {
        'x': x_traj_shifted,
        'y': y_traj_shifted,
        'z': z_traj,
        'color': params["color"],
        'length': len(x_traj_shifted)
    }
    
    # Initialize plots for animation
    line, = ax.plot([], [], [], '-', color=params["color"], linewidth=2, label=f'{rocket_name} Trajectory')
    point, = ax.plot([], [], [], 'o', color=params["color"], markersize=5)
    trajectory_lines[rocket_name] = line
    trajectory_points[rocket_name] = point

    # Find collision points with atmospheric layers
    collision_points_for_rocket = []
    for layer_name, layer_params in ATMOSPHERE_LAYERS.items():
        min_alt = layer_params["min_alt"]
        max_alt = layer_params["max_alt"]

        # Find points where trajectory enters/exits the layer
        # Entry: z_traj crosses min_alt from below
        # Exit: z_traj crosses max_alt from below
        
        # Check for entry into layer (from below min_alt to above min_alt)
        entry_indices = np.where((z_traj[:-1] < min_alt) & (z_traj[1:] >= min_alt))[0]
        for idx in entry_indices:
            # Interpolate to find precise intersection point
            if z_traj[idx+1] - z_traj[idx] != 0:
                fraction = (min_alt - z_traj[idx]) / (z_traj[idx+1] - z_traj[idx])
                coll_x = x_traj_shifted[idx] + fraction * (x_traj_shifted[idx+1] - x_traj_shifted[idx])
                coll_y = y_traj_shifted[idx] + fraction * (y_traj_shifted[idx+1] - y_traj_shifted[idx])
                coll_z = min_alt
                collision_points_for_rocket.append((coll_x, coll_y, coll_z, f'Entry {layer_name}'))
        
        # Check for exit from layer (from below max_alt to above max_alt)
        exit_indices = np.where((z_traj[:-1] < max_alt) & (z_traj[1:] >= max_alt))[0]
        for idx in exit_indices:
            if z_traj[idx+1] - z_traj[idx] != 0:
                fraction = (max_alt - z_traj[idx]) / (z_traj[idx+1] - z_traj[idx])
                coll_x = x_traj_shifted[idx] + fraction * (x_traj_shifted[idx+1] - x_traj_shifted[idx])
                coll_y = y_traj_shifted[idx] + fraction * (y_traj_shifted[idx+1] - y_traj_shifted[idx])
                coll_z = max_alt
                collision_points_for_rocket.append((coll_x, coll_y, coll_z, f'Exit {layer_name}'))

    collision_markers[rocket_name] = ax.plot([], [], [], 'X', color=params["color"], markersize=8, label=f'{rocket_name} Collisions')[0]
    
    # Store collision points for this rocket
    trajectory_data[rocket_name]['collisions'] = collision_points_for_rocket


# Set plot limits based on Nepal map and max altitude
max_z_overall = max(params['max_alt'] for params in ATMOSPHERE_LAYERS.values())
ax.set_xlim(NEPAL_X_MIN, NEPAL_X_MAX)
ax.set_ylim(NEPAL_Y_MIN, NEPAL_Y_MAX)
ax.set_zlim(0, max_z_overall * 1.1)

ax.set_xlabel('Longitude (m)')
ax.set_ylabel('Latitude (m)')
ax.set_zlabel('Altitude (m)')
ax.set_title('Rocket Trajectories through Atmospheric Layers over Nepal')
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

        # Update trajectory line
        trajectory_lines[rocket_name].set_data_3d(data['x'][:current_idx+1], data['y'][:current_idx+1], data['z'][:current_idx+1])
        artists.append(trajectory_lines[rocket_name])

        # Update rocket point
        trajectory_points[rocket_name].set_data_3d([data['x'][current_idx]], [data['y'][current_idx]], [data['z'][current_idx]])
        artists.append(trajectory_points[rocket_name])

        # Update collision markers (show all up to current frame)
        coll_x_display = []
        coll_y_display = []
        coll_z_display = []
        for coll_x, coll_y, coll_z, _ in data['collisions']:
            # Only show collision markers if the rocket has passed that point
            # Check if the current point's x and z coordinates are beyond the collision point's x and z.
            # This assumes a generally forward and upward trajectory.
            if current_idx > 0 and data['x'][current_idx] >= coll_x and data['z'][current_idx] >= coll_z:
                coll_x_display.append(coll_x)
                coll_y_display.append(coll_y)
                coll_z_display.append(coll_z)
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
