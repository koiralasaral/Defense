import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# --- 1. Global Configuration and Physical Constants ---
dt = 0.1  # Time step for the animation (seconds)
g = 9.81  # Acceleration due to gravity (m/s^2)

# --- 2. Atmospheric Layers (Simplified for 2D visualization) ---
# Altitudes are approximate and can vary.
# Source: NASA / NOAA typical atmosphere profiles
troposphere_alt = 12000  # 0-12 km (approx)
stratosphere_alt = 50000 # 12-50 km (approx)
mesosphere_alt = 85000   # 50-85 km (approx)
thermosphere_alt = 600000 # 85-600 km (beyond this is space)

# Colors for atmospheric layers
LAYER_COLORS = {
    'Troposphere': '#ADD8E6',   # Light blue
    'Stratosphere': '#87CEEB',  # Sky blue
    'Mesosphere': '#6495ED',    # Cornflower blue
    'Thermosphere': '#4682B4'   # Steel blue
}
LAYER_LABELS = {
    'Troposphere': 'Troposphere (0-12 km)',
    'Stratosphere': 'Stratosphere (12-50 km)',
    'Mesosphere': 'Mesosphere (50-85 km)',
    'Thermosphere': 'Thermosphere (85-600 km)'
}

# --- 3. Rocket Parameters (More realistic trajectory) ---
rocket_mass_initial = 10000  # kg (initial mass including fuel)
rocket_mass_dry = 2000       # kg (mass without fuel)
fuel_consumption_rate = 100  # kg/s (constant for simplicity during burn)
exhaust_velocity = 2500      # m/s (effective exhaust velocity)
rocket_thrust_force = fuel_consumption_rate * exhaust_velocity # Constant thrust while burning

rocket_burn_time = (rocket_mass_initial - rocket_mass_dry) / fuel_consumption_rate # Duration of engine burn
rocket_angle_deg = 80        # Launch angle from horizontal (degrees)
rocket_angle_rad = np.deg2rad(rocket_angle_deg)
rocket_pos = np.array([0.0, 0.0]) # Launch from (0,0) - our hypothetical launch site
rocket_vel = np.array([0.0, 0.0])
rocket_mass = rocket_mass_initial

# --- 4. Trailing Projectile Parameters ---
projectile_detach_time = 25.0 # seconds (after rocket launch)
projectile_launched = False
projectile_pos = np.array([0.0, 0.0])
projectile_vel = np.array([0.0, 0.0])
projectile_mass = 100 # kg

# --- 5. Enemy Missile Parameters ---
# This missile will appear high up and travel towards the rocket/projectile's path
enemy_missile_start_x = 100000.0 # meters (far right) - Changed to float
enemy_missile_start_y = 150000.0 # meters (high altitude) - Changed to float
enemy_missile_speed = 500 # m/s (very fast)
enemy_missile_pos = np.array([enemy_missile_start_x, enemy_missile_start_y]) # Now correctly initialized as float array

# Target a point where the rocket might be (simplified targeting)
enemy_target_x = 50000.0 # Changed to float
enemy_target_y = 100000.0 # Changed to float
enemy_missile_dir = (np.array([enemy_target_x, enemy_target_y]) - enemy_missile_pos)
enemy_missile_dir = enemy_missile_dir / np.linalg.norm(enemy_missile_dir)
enemy_missile_vel = enemy_missile_dir * enemy_missile_speed

# --- 6. Interceptor Parameters (Launched from "Nepal Ground Defense") ---
interceptor_launch_x = 10000.0 # 10 km from rocket launch, for visual separation - Changed to float
interceptor_launch_y = 0.0
interceptor_speed = 700 # m/s (faster than enemy)
interceptor_launched = False
interceptor_pos = np.array([interceptor_launch_x, interceptor_launch_y])
interceptor_vel = np.array([0.0, 0.0])
interceptor_launch_detection_time = 30.0 # Time when enemy missile is detected and interceptor readies
interceptor_launch_delay = 5.0 # Delay after detection for actual launch

# --- 7. Detection and Interception thresholds ---
intercept_distance_threshold = 1000 # meters (max distance for hit)
enemy_missile_intercepted = False

# --- 8. Plot Limits (Dynamic, adjust based on expected max altitude/range) ---
# These need to be large enough to contain the trajectory
x_max_plot = 200000 # 200 km horizontal range
y_max_plot = 200000 # 200 km altitude

# --- 9. Data Storage for Trails ---
rocket_trail = []
projectile_trail = []
enemy_missile_trail = []
interceptor_trail = []

# --- 10. Setup the Matplotlib Plot ---
fig, ax = plt.subplots(figsize=(15, 10))
ax.set_xlim(0, x_max_plot)
ax.set_ylim(0, y_max_plot)
ax.set_aspect('equal', adjustable='box')
ax.set_title("Rocket Launch & Missile Interception over Nepal (2D Simulation)")
ax.set_xlabel("Horizontal Distance (m)")
ax.set_ylabel("Altitude (m)")
ax.grid(True, linestyle='--', alpha=0.7)

# Add atmospheric layers as shaded regions
ax.axhspan(0, troposphere_alt, facecolor=LAYER_COLORS['Troposphere'], alpha=0.5, label=LAYER_LABELS['Troposphere'])
ax.axhspan(troposphere_alt, stratosphere_alt, facecolor=LAYER_COLORS['Stratosphere'], alpha=0.5, label=LAYER_LABELS['Stratosphere'])
ax.axhspan(stratosphere_alt, mesosphere_alt, facecolor=LAYER_COLORS['Mesosphere'], alpha=0.5, label=LAYER_LABELS['Mesosphere'])
ax.axhspan(mesosphere_alt, thermosphere_alt, facecolor=LAYER_COLORS['Thermosphere'], alpha=0.5, label=LAYER_LABELS['Thermosphere'])

# Nepal ground representation (simplified as a green line)
ax.plot([0, x_max_plot], [0, 0], color='green', linewidth=4, label='Nepal Ground')
ax.text(x_max_plot * 0.45, 0.02 * y_max_plot, "Nepal", fontsize=14, color='darkgreen', ha='center')


# Plot elements (initially empty)
rocket_line, = ax.plot([], [], 'o-', color='blue', markersize=8, label='Rocket')
projectile_line, = ax.plot([], [], 'o', color='green', markersize=6, label='Payload')
enemy_missile_line, = ax.plot([], [], 'x', color='red', markersize=10, label='Enemy Missile')
interceptor_line, = ax.plot([], [], '^', color='purple', markersize=8, label='Interceptor')

# Trails
rocket_trail_plot, = ax.plot([], [], '--', color='lightblue', alpha=0.6)
projectile_trail_plot, = ax.plot([], [], ':', color='lightgreen', alpha=0.6)
enemy_missile_trail_plot, = ax.plot([], [], ':', color='pink', alpha=0.6)
interceptor_trail_plot, = ax.plot([], [], ':', color='violet', alpha=0.6)

# Labels for defense sites
ax.plot(0, 0, 's', color='darkblue', markersize=12, label='Rocket Launch Site')
ax.text(0, -5000, 'Rocket Launch Site', color='darkblue', fontsize=10, ha='left')

ax.plot(interceptor_launch_x, interceptor_launch_y, 's', color='darkgreen', markersize=12, label='Defense System')
ax.text(interceptor_launch_x, -5000, 'Defense System', color='darkgreen', fontsize=10, ha='center')


time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12)
status_text = ax.text(0.02, 0.94, '', transform=ax.transAxes, fontsize=12, color='darkred')
altitude_text = ax.text(0.02, 0.90, '', transform=ax.transAxes, fontsize=12, color='blue')

ax.legend(loc='lower right', fancybox=True, shadow=True, ncol=2)

# --- 11. Simulation State Variables ---
current_time = 0.0
enemy_missile_intercepted = False
projectile_landed = False
rocket_engine_on = True
interceptor_launch_event_fired = False


# --- 12. Animation Update Function ---
def update(frame):
    global current_time, projectile_launched, projectile_pos, projectile_vel, \
           interceptor_launched, interceptor_pos, interceptor_vel, \
           enemy_missile_intercepted, rocket_pos, rocket_vel, rocket_mass, \
           rocket_engine_on, projectile_landed, interceptor_launch_event_fired, \
           enemy_missile_pos, enemy_missile_vel # Added enemy_missile_pos and enemy_missile_vel to global

    # Stop condition
    if enemy_missile_intercepted or (projectile_landed and rocket_pos[1] < 1000): # Stop if both landed/intercepted
        return [] # Return empty list if no updates

    # --- Update Rocket Physics ---
    if rocket_engine_on:
        if current_time <= rocket_burn_time:
            # Thrust force components
            thrust_x = rocket_thrust_force * np.cos(rocket_angle_rad)
            thrust_y = rocket_thrust_force * np.sin(rocket_angle_rad)
            # Acceleration from thrust
            accel_thrust_x = thrust_x / rocket_mass
            accel_thrust_y = thrust_y / rocket_mass
            # Apply acceleration
            rocket_vel[0] += accel_thrust_x * dt
            rocket_vel[1] += accel_thrust_y * dt
            # Mass reduction
            rocket_mass -= fuel_consumption_rate * dt
        else:
            rocket_engine_on = False # Engine cut-off
            # print(f"Rocket engine cut-off at {current_time:.2f}s. Mass: {rocket_mass:.2f} kg")

    # Apply gravity to rocket (always)
    rocket_vel[1] -= g * dt
    rocket_pos += rocket_vel * dt
    rocket_trail.append(rocket_pos.copy())

    # Ensure rocket doesn't go below ground (if it somehow dips during simulation setup)
    if rocket_pos[1] < 0:
        rocket_pos[1] = 0
        rocket_vel[1] = 0 # Stop vertical movement
        if rocket_engine_on: # If engine was still on, turn off
            rocket_engine_on = False

    # --- Update Trailing Projectile (Payload) ---
    if not projectile_launched and current_time >= projectile_detach_time:
        projectile_launched = True
        # Projectile detaches with rocket's current velocity
        projectile_pos = rocket_pos.copy()
        projectile_vel = rocket_vel.copy()
        # Add a slight "kick" for separation or just inherit velocity
        # projectile_vel += np.array([-5.0, 0.0]) # Example of a kick

    if projectile_launched and not projectile_landed:
        projectile_vel[1] -= g * dt # Apply gravity
        projectile_pos += projectile_vel * dt
        projectile_trail.append(projectile_pos.copy())
        if projectile_pos[1] <= 0: # Check if projectile hit ground
            projectile_pos[1] = 0
            projectile_vel = np.array([0.0, 0.0]) # Stop movement
            projectile_landed = True
            status_text.set_text(f'Status: Payload Landed at ({projectile_pos[0]/1000:.1f} km, {projectile_pos[1]/1000:.1f} km)')

    # --- Update Enemy Missile ---
    if not enemy_missile_intercepted:
        enemy_missile_pos += enemy_missile_vel * dt
        enemy_missile_trail.append(enemy_missile_pos.copy())

    # --- Update Interceptor ---
    if not interceptor_launched and not interceptor_launch_event_fired and current_time >= interceptor_launch_detection_time:
        interceptor_launch_event_fired = True # Mark that we've "detected"
        # Calculate interceptor's initial velocity to target enemy missile's CURRENT position
        # For a more realistic scenario, this would predict the enemy's future position.
        # But for simplicity, we just aim directly at current target
        target_dir = (enemy_missile_pos - interceptor_pos)
        interceptor_dir_norm = np.linalg.norm(target_dir)
        if interceptor_dir_norm > 0:
            interceptor_vel = (target_dir / interceptor_dir_norm) * interceptor_speed
        else:
            interceptor_vel = np.array([0.0, 0.0]) # Avoid division by zero

    if interceptor_launched and not enemy_missile_intercepted:
        interceptor_pos += interceptor_vel * dt
        interceptor_trail.append(interceptor_pos.copy())

    # Actual launch happens after a delay from detection
    if interceptor_launch_event_fired and not interceptor_launched and \
       current_time >= (interceptor_launch_detection_time + interceptor_launch_delay):
        interceptor_launched = True
        # Interceptor starts moving from its launch site
        interceptor_pos = np.array([interceptor_launch_x, interceptor_launch_y]) # Reset to launch point for visual


    # --- Check for Interception ---
    if not enemy_missile_intercepted and interceptor_launched:
        distance_to_intercept = np.linalg.norm(interceptor_pos - enemy_missile_pos)
        if distance_to_intercept < intercept_distance_threshold:
            enemy_missile_intercepted = True
            status_text.set_text(f'Status: ENEMY MISSILE INTERCEPTED at Time: {current_time:.2f}s!')
            print(f'INTERCEPTION SUCCESS! Time: {current_time:.2f}s, Altitude: {interceptor_pos[1]/1000:.1f} km')


    # --- Update Plot Elements ---
    # Rocket
    if rocket_pos[1] > 0 or rocket_engine_on: # Keep plotting rocket until it's very low or engine off
        rocket_line.set_data([rocket_pos[0]], [rocket_pos[1]])
        rocket_trail_plot.set_data([p[0] for p in rocket_trail], [p[1] for p in rocket_trail])
    else: # Hide rocket if it's "landed" after engine cut-off
        rocket_line.set_data([], [])
        rocket_trail_plot.set_data([], [])

    # Projectile
    if projectile_launched and not projectile_landed:
        projectile_line.set_data([projectile_pos[0]], [projectile_pos[1]])
        projectile_trail_plot.set_data([p[0] for p in projectile_trail], [p[1] for p in projectile_trail])
    else:
        projectile_line.set_data([], [])
        projectile_trail_plot.set_data([], [])

    # Enemy Missile
    if not enemy_missile_intercepted:
        enemy_missile_line.set_data([enemy_missile_pos[0]], [enemy_missile_pos[1]])
        enemy_missile_trail_plot.set_data([p[0] for p in enemy_missile_trail], [p[1] for p in enemy_missile_trail])
    else:
        enemy_missile_line.set_data([], [])
        enemy_missile_trail_plot.set_data([], [])

    # Interceptor
    if interceptor_launched and not enemy_missile_intercepted:
        interceptor_line.set_data([interceptor_pos[0]], [interceptor_pos[1]])
        interceptor_trail_plot.set_data([p[0] for p in interceptor_trail], [p[1] for p in interceptor_trail])
    else:
        interceptor_line.set_data([], [])
        interceptor_trail_plot.set_data([], [])

    # Update Text Displays
    time_text.set_text(f'Time: {current_time:.2f} s')
    altitude_text.set_text(f'Rocket Alt: {rocket_pos[1]/1000:.1f} km, Speed: {np.linalg.norm(rocket_vel):.1f} m/s')
    if not enemy_missile_intercepted and not status_text.get_text().startswith('Status: ENEMY MISSILE INTERCEPTED'):
        status_text.set_text(f'Status: Enemy Missile Incoming ({np.linalg.norm(enemy_missile_pos - interceptor_pos)/1000:.1f} km dist)')
    elif enemy_missile_intercepted:
        pass # Keep interception message
    elif projectile_landed:
        pass # Keep payload landed message


    current_time += dt

    return rocket_line, projectile_line, enemy_missile_line, interceptor_line, \
           rocket_trail_plot, projectile_trail_plot, enemy_missile_trail_plot, \
           interceptor_trail_plot, time_text, status_text, altitude_text


# Create the animation
# frames: Adjust as needed. Higher frames = longer animation.
# interval: ms between frames. dt * 1000 is for real-time speed.
ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 1000),
                              interval=dt * 1000, blit=True, repeat=False)

plt.show()

# Uncomment to save the animation (requires ffmpeg or imagemagick)
print("Saving animation... this may take a while.")
ani.save('rocket_interception_nepal.gif', writer='pillow', fps=10) # fps affects speed of saved gif
ani.save('rocket_interception_nepal.mp4', writer='ffmpeg', fps=30, dpi=200)
