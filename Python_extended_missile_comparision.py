import folium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from folium.plugins import TimestampedGeoJson
import math, random, datetime
from mpl_toolkits.mplot3d import Axes3D  # For optional 3D plotting


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great–circle distance (in km) between two (lat,lon) points.
    """
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a Fourier-perturbed point along the line connecting start and end.

    Parameters:
      t      : float in [0,1] sample parameter.
      start  : tuple (lat, lon) for the start.
      end    : tuple (lat, lon) for the end.
      A, B   : amplitude multipliers for latitude and longitude variations.
      k      : frequency multiplier.
      phase  : phase shift.

    Returns:
      (lat, lon) tuple.
    """
    lat_lin = start[0] + t * (end[0] - start[0])
    lon_lin = start[1] + t * (end[1] - start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)


def taylor_approximation(t0, t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute the first-order Taylor series approximation of the Fourier curve
    around t = t0.

    f(t) ~ f(t0) + f'(t0) * (t - t0)

    Returns approximate (lat, lon) tuple.
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0 + delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return tuple(p0 + deriv * (t - t0))


# =============================================================================
# USER-DEFINED PARAMETERS & DATA
# =============================================================================
# We define our two capitals.
capitals = {
    "USA": (38.9072, -77.0369),  # Washington, DC
    "Russia": (55.7558, 37.6173)  # Moscow
}

# =============================================================================
# SIMULATE BATTLEFRONT & DEPLOYMENT POINTS
# =============================================================================
print("Computing battlefront deployment points...")

# Use 10 deployment points along a Fourier-perturbed border.
n_points = 10
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3) for t in t_values]

# For this simulation, alternate deployment ownership:
deployment_data = []
for i, pt in enumerate(curve_points):
    if i % 2 == 0:
        country = "USA"
    else:
        country = "Russia"
    if country == "USA":
        op_range = 160
        cost_fac = 4
        dist = haversine(capitals["USA"][0], capitals["USA"][1], pt[0], pt[1])
    else:
        op_range = 30
        cost_fac = 3
        dist = haversine(capitals["Russia"][0], capitals["Russia"][1], pt[0], pt[1])
    req = math.ceil((dist / op_range) * cost_fac)
    deployment_data.append({
        "lat": pt[0],
        "lon": pt[1],
        "country": country,
        "required": req
    })

print("Deployment Data:")
for i, d in enumerate(deployment_data):
    print(f"  Point {i}: lat={d['lat']:.4f}, lon={d['lon']:.4f}, country={d['country']}, required={d['required']}")

# =============================================================================
# SIMULATE MISSILE LAUNCHES FROM EACH DEPLOYMENT POINT
# =============================================================================
# For each deployment point, simulate a missile launch.
# For USA, missile target = launch + 0.5*(Russia_cap - USA_cap)
# For Russia, missile target = launch - 0.5*(Russia_cap - USA_cap)
USA_cap = capitals["USA"]
Russia_cap = capitals["Russia"]
v = (Russia_cap[0] - USA_cap[0], Russia_cap[1] - USA_cap[1])
factor = 0.5

# Use n_samples = 11 (indices 0 to 10) to produce a discrete missile trajectory.
missile_launches = []
for d in deployment_data:
    launch = (d["lat"], d["lon"])
    if d["country"] == "USA":
        target = (launch[0] + factor * v[0], launch[1] + factor * v[1])
    else:
        target = (launch[0] - factor * v[0], launch[1] - factor * v[1])
    t_samples = np.linspace(0, 1, 11)
    trajectory_fourier = [fourier_curve(t, launch, target, A=0.5, B=0.5, k=2) for t in t_samples]
    trajectory_taylor = [taylor_approximation(0.5, t, launch, target, A=0.5, B=0.5, k=2) for t in t_samples]
    missile_launches.append({
        "launch": launch,
        "target": target,
        "country": d["country"],
        "trajectory_fourier": trajectory_fourier,
        "trajectory_taylor": trajectory_taylor
    })

# =============================================================================
# CREATE TIME-STAMPED ANIMATION IN FOLIUM FOR MISSILE LAUNCHES (2D)
# =============================================================================
print("Creating time-stamped animation for missile launches (2D)...")

# Compute center of map as the midpoint between the capitals.
center_lat = (capitals["USA"][0] + capitals["Russia"][0]) / 2
center_lon = (capitals["USA"][1] + capitals["Russia"][1]) / 2

missile_features = []
base_time = datetime.datetime.now()
for i, launch in enumerate(missile_launches):
    time_stamp = (base_time + datetime.timedelta(minutes=5 * i)).isoformat()
    # Fourier trajectory feature.
    fourier_coords = [[pt[1], pt[0]] for pt in launch["trajectory_fourier"]]
    feature_fourier = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": f"Missile {i + 1} ({launch['country']})<br>Launch: {launch['launch']}<br>Target: {launch['target']}<br>(Fourier)",
            "style": {"color": "green" if launch["country"] == "USA" else "orange", "weight": 3}
        },
        "geometry": {
            "type": "LineString",
            "coordinates": fourier_coords
        }
    }
    # Taylor trajectory feature.
    taylor_coords = [[pt[1], pt[0]] for pt in launch["trajectory_taylor"]]
    feature_taylor = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": f"Missile {i + 1} ({launch['country']})<br>Launch: {launch['launch']}<br>Target: {launch['target']}<br>(Taylor)",
            "style": {"color": "blue" if launch["country"] == "USA" else "red", "weight": 3, "dashArray": "5,5"}
        },
        "geometry": {
            "type": "LineString",
            "coordinates": taylor_coords
        }
    }
    missile_features.append(feature_fourier)
    missile_features.append(feature_taylor)

missile_geojson = {"type": "FeatureCollection", "features": missile_features}
missile_time_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
popup1 = folium.LatLngPopup()

missile_time_map.add_child(popup1)
TimestampedGeoJson(
    data=missile_geojson,
    period="PT5M",  # Each missile launch appears every 5 minutes
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(missile_time_map)
# Optionally add the battlefront (deployment border) for reference.
from folium import PolyLine

PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(missile_time_map)
missile_time_map.save("missile_launch_timed.html")
print("Missile launch animated map saved as 'missile_launch_timed.html'.")

# =============================================================================
# CREATE MATPLOTLIB SUBPLOTS FOR EACH MISSILE LAUNCH TRAJECTORY (2D)
# =============================================================================
print("Creating Matplotlib subplots for each missile launch trajectory...")

num_missiles = len(missile_launches)
cols = 2
rows = math.ceil(num_missiles / cols)
fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 4))
axes = axes.flatten()

for i, launch in enumerate(missile_launches):
    ax = axes[i]
    # Fourier trajectory coordinates.
    f_traj = launch["trajectory_fourier"]
    f_lats = [pt[0] for pt in f_traj]
    f_lons = [pt[1] for pt in f_traj]
    # Taylor trajectory coordinates.
    t_traj = launch["trajectory_taylor"]
    t_lats = [pt[0] for pt in t_traj]
    t_lons = [pt[1] for pt in t_traj]

    if launch["country"] == "USA":
        ax.plot(f_lons, f_lats, '-', color="green", label="USA Fourier")
        ax.plot(t_lons, t_lats, '--', color="blue", label="USA Taylor")
        ax.scatter(launch["launch"][1], launch["launch"][0], c="black", marker="o", s=50, label="Launch")
        ax.scatter(launch["target"][1], launch["target"][0], c="purple", marker="x", s=50, label="Target")
    else:
        ax.plot(f_lons, f_lats, '-', color="orange", label="Russia Fourier")
        ax.plot(t_lons, t_lats, '--', color="red", label="Russia Taylor")
        ax.scatter(launch["launch"][1], launch["launch"][0], c="black", marker="o", s=50, label="Launch")
        ax.scatter(launch["target"][1], launch["target"][0], c="purple", marker="x", s=50, label="Target")

    ax.set_title(f"Missile {i + 1} ({launch['country']})")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    # Ensure that each subplot displays a legend once.
    leg_labels = ax.get_legend_handles_labels()[1]
    if len(leg_labels) == 0:
        ax.legend()

# Hide any extra empty subplots.
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.savefig("missile_trajectories_subplots.png")
plt.show()
print("Matplotlib subplots saved as 'missile_trajectories_subplots.png'.")

# =============================================================================
# OPTIONAL: CREATE 3D ANIMATION OF PROJECTILE TRAJECTORIES
# =============================================================================
print("Creating 3D projectile animation...")
v_proj = 100.0
theta_proj = np.radians(45)
g = 9.81
flight_time = 2 * v_proj * np.sin(theta_proj) / g


def compute_3d_trajectory(launch, target, n_samples=11, v_proj=100, theta_proj=np.radians(45), g=9.81):
    """
    Compute a 3D projectile trajectory.
    Horizontal (x, y): linear interpolation between launch and target.
    Vertical (z): computed using projectile motion.
    """
    t_samples = np.linspace(0, 1, n_samples)
    trajectory = []
    for t in t_samples:
        x = launch[1] + (target[1] - launch[1]) * t
        y = launch[0] + (target[0] - launch[0]) * t
        t_actual = t * flight_time
        z = v_proj * np.sin(theta_proj) * t_actual - 0.5 * g * (t_actual ** 2)
        trajectory.append((x, y, z))
    return trajectory


missile_3d_trajectories = []
for launch in missile_launches:
    traj3d = compute_3d_trajectory(launch["launch"], launch["target"], n_samples=11, v_proj=v_proj,
                                   theta_proj=theta_proj, g=g)
    missile_3d_trajectories.append({"trajectory": traj3d, "country": launch["country"]})

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Projectile Trajectories")

all_x, all_y, all_z = [], [], []
for traj in missile_3d_trajectories:
    for pt in traj["trajectory"]:
        all_x.append(pt[0])
        all_y.append(pt[1])
        all_z.append(pt[2])
ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.5)
ax.set_zlim(0, max(all_z) + 20)

lines_3d = []
colors = {"USA": "green", "Russia": "orange"}
for traj in missile_3d_trajectories:
    line, = ax.plot([], [], [], marker="o", color=colors.get(traj["country"], "black"), label=traj["country"])
    lines_3d.append(line)


def init_3d():
    for line in lines_3d:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines_3d


def update_3d(frame):
    for idx, traj in enumerate(missile_3d_trajectories):
        current_traj = traj["trajectory"][:frame + 1]
        xs = [pt[0] for pt in current_traj]
        ys = [pt[1] for pt in current_traj]
        zs = [pt[2] for pt in current_traj]
        lines_3d[idx].set_data(xs, ys)
        lines_3d[idx].set_3d_properties(zs)
    return lines_3d


ani = FuncAnimation(fig, update_3d, frames=11, init_func=init_3d, interval=500, blit=False, repeat=False)
plt.tight_layout()
plt.show()
print("3D projectile animation complete.")

# =============================================================================
# END OF SIMULATION
# =============================================================================
print("\nAll simulations executed. Check the generated HTML, PNG, and animation windows for results.")