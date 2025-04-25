import folium
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from folium.plugins import TimestampedGeoJson
import math, random, datetime
from mpl_toolkits.mplot3d import Axes3D  # required for 3D plotting

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
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a perturbed point along the line from start to end.
    Uses a Fourier (sinusoidal) perturbation added to the linear interpolation.
    """
    lat_lin = start[0] + t*(end[0] - start[0])
    lon_lin = start[1] + t*(end[1] - start[1])
    lat_pert = A * np.sin(2*np.pi*k*t + phase)
    lon_pert = B * np.cos(2*np.pi*k*t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)

def taylor_approximation(t0, t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a first–order Taylor series approximation of the Fourier curve about t=t0.
    f(t) ~ f(t0) + f'(t0) * (t - t0)
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0 + delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return tuple(p0 + deriv * (t - t0))

# =============================================================================
# USER-DEFINED PARAMETERS & DATA
# =============================================================================
# Capitals (used for defining the battlefront)
capitals = {
    "USA": (38.9072, -77.0369),   # Washington, DC
    "Russia": (55.7558, 37.6173)   # Moscow
}

# =============================================================================
# SIMULATE BATTLEFRONT & DEPLOYMENT POINTS
# =============================================================================
print("Computing battlefront deployment points...")

n_points = 10  # Use 10 deployment points (for 10 weapon systems)
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3) for t in t_values]

# For illustration, alternate deployment ownership:
deployment_data = []
for i, pt in enumerate(curve_points):
    if i % 2 == 0:
        country = "USA"
    else:
        country = "Russia"
    # Define a dummy 'required' value (for context)
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
# SIMULATE MISSILE (PROJECTILE) LAUNCHES FROM EACH DEPLOYMENT POINT:
# For each deployment point, launch a missile toward the target.
# For USA deployments, target = launch + 0.5*(Russia_cap - USA_cap)
# For Russia deployments, target = launch - 0.5*(Russia_cap - USA_cap)
USA_cap = capitals["USA"]
Russia_cap = capitals["Russia"]
v = (Russia_cap[0] - USA_cap[0], Russia_cap[1] - USA_cap[1])
factor = 0.5

# We'll use n_samples = 11 so that sample indices run from 0 to 10.
missile_launches = []
for d in deployment_data:
    launch = (d["lat"], d["lon"])
    if d["country"] == "USA":
        target = (launch[0] + factor * v[0], launch[1] + factor * v[1])
    else:
        target = (launch[0] - factor * v[0], launch[1] - factor * v[1])
    t_samples = np.linspace(0, 1, 11)  # 11 sample points: n = 0 to 10
    # Compute two trajectories: via Fourier and via Taylor (for demonstration)
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

# Center the map as the midpoint between the capitals.
center_lat = (capitals["USA"][0] + capitals["Russia"][0]) / 2
center_lon = (capitals["USA"][1] + capitals["Russia"][1]) / 2

missile_features = []
base_time = datetime.datetime.now()
for i, launch in enumerate(missile_launches):
    time_stamp = (base_time + datetime.timedelta(minutes=5*i)).isoformat()
    # Fourier trajectory as a LineString (GeoJSON requires [lon, lat])
    fourier_coords = [[pt[1], pt[0]] for pt in launch["trajectory_fourier"]]
    feature_fourier = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": f"{launch['country']} missile<br>Launch: {launch['launch']}<br>Target: {launch['target']}<br>(Fourier trajectory)",
            "style": {"color": "green" if launch["country"]=="USA" else "orange", "weight": 3}
        },
        "geometry": {
            "type": "LineString",
            "coordinates": fourier_coords
        }
    }
    # Taylor trajectory as a LineString
    taylor_coords = [[pt[1], pt[0]] for pt in launch["trajectory_taylor"]]
    feature_taylor = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": f"{launch['country']} missile<br>Launch: {launch['launch']}<br>Target: {launch['target']}<br>(Taylor trajectory)",
            "style": {"color": "blue" if launch["country"]=="USA" else "red", "weight": 3, "dashArray": "5,5"}
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
TimestampedGeoJson(
    data=missile_geojson,
    period="PT5M",  # Each missile launch appears 5 minutes apart
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(missile_time_map)
# Optionally add battlefront for reference.
from folium import PolyLine
PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(missile_time_map)
missile_time_map.save("missile_launch_timed.html")
print("Missile launch 2D animated map saved as 'missile_launch_timed.html'.")

# =============================================================================
# PROGRAM 6: 3D PROJECTILE ANIMATION USING MATPLOTLIB
# =============================================================================
print("Creating 3D projectile animation...")

# We'll build a 3D trajectory for each missile using a simple projectile model.
# For this, we assume that the horizontal motion follows a linear interpolation from launch to target,
# and the vertical (altitude) component is computed using projectile motion.
# We'll use:
#   v_proj = 100 m/s (initial speed)
#   theta_proj = 45° (launch angle)
#   g = 9.81 m/s^2
# Compute flight time:
v_proj = 100.0
theta_proj = np.radians(45)
g = 9.81
flight_time = 2 * v_proj * np.sin(theta_proj) / g

def compute_3d_trajectory(launch, target, n_samples=11, v_proj=100, theta_proj=np.radians(45), g=9.81):
    """
    Compute a 3D projectile trajectory.
    Horizontal coordinates: linear interpolation between launch and target.
    Vertical coordinates (altitude): computed via projectile motion.
    """
    t_samples = np.linspace(0, 1, n_samples)
    trajectory = []
    for t in t_samples:
        # Horizontal (x, y): use linear interpolation.
        # Here we assume x corresponds to longitude and y to latitude.
        x = launch[1] + (target[1] - launch[1]) * t
        y = launch[0] + (target[0] - launch[0]) * t
        # Map parameter t to actual time
        time_actual = t * flight_time
        # Projectile motion formula for altitude (z):
        z = v_proj * np.sin(theta_proj) * time_actual - 0.5 * g * (time_actual ** 2)
        trajectory.append((x, y, z))
    return trajectory

# Compute 3D trajectories for all missile launches.
missile_3d_trajectories = []
for launch in missile_launches:
    traj3d = compute_3d_trajectory(launch["launch"], launch["target"], n_samples=11, v_proj=v_proj, theta_proj=theta_proj, g=g)
    missile_3d_trajectories.append({"trajectory": traj3d, "country": launch["country"]})

# 3D Animation using matplotlib
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_zlabel("Altitude (m)")
ax.set_title("3D Projectile Trajectories")

# Set dynamic limits based on our simulated data.
all_x = []
all_y = []
all_z = []
for traj in missile_3d_trajectories:
    for pt in traj["trajectory"]:
        all_x.append(pt[0])
        all_y.append(pt[1])
        all_z.append(pt[2])
ax.set_xlim(min(all_x)-0.5, max(all_x)+0.5)
ax.set_ylim(min(all_y)-0.5, max(all_y)+0.5)
ax.set_zlim(0, max(all_z)+20)

# Prepare a list to collect line objects for each missile.
lines = []
colors = {"USA": "green", "Russia": "orange"}
for traj in missile_3d_trajectories:
    line, = ax.plot([], [], [], marker="o", color=colors.get(traj["country"], "black"), label=traj["country"])
    lines.append(line)

def init():
    for line in lines:
        line.set_data([], [])
        line.set_3d_properties([])
    return lines

def update(frame):
    # frame will go from 0 to n_samples (here 11)
    for idx, traj in enumerate(missile_3d_trajectories):
        current_traj = traj["trajectory"][:frame+1]
        xs = [pt[0] for pt in current_traj]
        ys = [pt[1] for pt in current_traj]
        zs = [pt[2] for pt in current_traj]
        lines[idx].set_data(xs, ys)
        lines[idx].set_3d_properties(zs)
    return lines

ani = FuncAnimation(fig, update, frames=11, init_func=init, interval=500, blit=False, repeat=False)
plt.tight_layout()
plt.show()
print("3D projectile animation complete.")