import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import TimestampedGeoJson
import math, random, datetime

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great-circle distance (in km) between two geographic points.
    """
    R = 6371  # Radius of the Earth in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a Fourier-perturbed point along a straight line between start and end.
    """
    lat_lin = start[0] + t * (end[0] - start[0])
    lon_lin = start[1] + t * (end[1] - start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)

def taylor_approximation(t0, t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute the first-order Taylor series approximation of the Fourier curve about t=t0.
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0+delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return tuple(p0 + deriv * (t - t0))

# =============================================================================
# USER-DEFINED PARAMETERS & DATA
# =============================================================================
# Define the capitals (endpoints)
capitals = {
    "USA": (38.9072, -77.0369),   # Washington, DC
    "Russia": (55.7558, 37.6173)   # Moscow
}

# =============================================================================
# SIMULATE BATTLEFRONT & DEPLOYMENT POINTS
# =============================================================================
print("Computing battlefront deployment points...")
n_points = 10  # We deploy 10 weapons (one per deployment point) along the border
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3) for t in t_values]

# For simplicity, we alternate deployment ownership: even-index points belong to USA; odd-index to Russia.
deployment_data = []
for i, pt in enumerate(curve_points):
    if i % 2 == 0:
        country = "USA"
    else:
        country = "Russia"
    # Compute a required number (here, use a dummy cost formula based on distance)
    if country == "USA":
        op_range = 160
        cost_fac = 4
        dist = haversine(capitals["USA"][0], capitals["USA"][1], pt[0], pt[1])
    else:
        op_range = 30
        cost_fac = 3
        dist = haversine(capitals["Russia"][0], capitals["Russia"][1], pt[0], pt[1])
    required = math.ceil((dist / op_range) * cost_fac)
    deployment_data.append({
        "lat": pt[0],
        "lon": pt[1],
        "country": country,
        "required": required
    })

print("Deployment Data:")
for i, d in enumerate(deployment_data):
    print(f"  Point {i}: lat={d['lat']:.4f}, lon={d['lon']:.4f}, country={d['country']}, required={d['required']}")

# =============================================================================
# SIMULATE MISSILE LAUNCHES (PROJECTILES) FROM EACH DEPLOYMENT POINT
# =============================================================================
# Each deployment point launches a missile. Define target using a scaled vector.
USA_cap = capitals["USA"]
Russia_cap = capitals["Russia"]
v = (Russia_cap[0] - USA_cap[0], Russia_cap[1] - USA_cap[1])
factor = 0.5  # target is halfway in the direction vector

missile_launches = []
for d in deployment_data:
    launch = (d["lat"], d["lon"])
    if d["country"] == "USA":
        target = (launch[0] + factor * v[0], launch[1] + factor * v[1])
    else:
        target = (launch[0] - factor * v[0], launch[1] - factor * v[1])
    # Compute trajectories for missile launch (projectile trajectory)
    t_samples = np.linspace(0, 1, 20)  # 20 sample points along trajectory
    # Fourier trajectory with a small perturbation for missile path
    trajectory_fourier = [fourier_curve(t, launch, target, A=0.5, B=0.5, k=2) for t in t_samples]
    # Taylor approximation (first-order linearization) of the trajectory
    trajectory_taylor = [taylor_approximation(0.5, t, launch, target, A=0.5, B=0.5, k=2) for t in t_samples]
    missile_launches.append({
        "launch": launch,
        "target": target,
        "country": d["country"],
        "trajectory_fourier": trajectory_fourier,
        "trajectory_taylor": trajectory_taylor
    })

# =============================================================================
# CREATE TIME-STAMPED ANIMATION IN FOLIUM FOR PROJECTILE LAUNCHES
# =============================================================================
print("Creating time-stamped animation for missile launches...")

# Center map as midpoint between capitals.
center_lat = (capitals["USA"][0] + capitals["Russia"][0]) / 2
center_lon = (capitals["USA"][1] + capitals["Russia"][1]) / 2

missile_features = []
base_time = datetime.datetime.now()
for i, launch in enumerate(missile_launches):
    time_stamp = (base_time + datetime.timedelta(minutes=5*i)).isoformat()
    # Create a feature for the Fourier trajectory.
    fourier_coords = [[pt[1], pt[0]] for pt in launch["trajectory_fourier"]]  # GeoJSON requires [lon, lat]
    feature_fourier = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": (f"{launch['country']} missile<br>"
                      f"Launch: {launch['launch']}, Target: {launch['target']}<br>"
                      "Trajectory: Fourier"),
            "style": {"color": "green" if launch["country"] == "USA" else "orange", "weight": 3}
        },
        "geometry": {
            "type": "LineString",
            "coordinates": fourier_coords
        }
    }
    # Create a feature for the Taylor trajectory.
    taylor_coords = [[pt[1], pt[0]] for pt in launch["trajectory_taylor"]]
    feature_taylor = {
        "type": "Feature",
        "properties": {
            "time": time_stamp,
            "popup": (f"{launch['country']} missile<br>"
                      f"Launch: {launch['launch']}, Target: {launch['target']}<br>"
                      "Trajectory: Taylor"),
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
# Also add the battlefront for reference.
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(missile_time_map)
missile_time_map.save("missile_launch_timed.html")
print("Missile launch animated map saved as 'missile_launch_timed.html'.")

# =============================================================================
# CREATE MATPLOTLIB STATIC PLOT FOR MISSILE TRAJECTORIES
# =============================================================================
plt.figure(figsize=(10,6))
for launch in missile_launches:
    # Fourier trajectory
    f_traj = launch["trajectory_fourier"]
    f_lats = [pt[0] for pt in f_traj]
    f_lons = [pt[1] for pt in f_traj]
    # Taylor trajectory
    t_traj = launch["trajectory_taylor"]
    t_lats = [pt[0] for pt in t_traj]
    t_lons = [pt[1] for pt in t_traj]
    if launch["country"] == "USA":
        plt.plot(f_lons, f_lats, '-', color="green", label="USA Fourier" if "USA Fourier" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(t_lons, t_lats, '--', color="blue", label="USA Taylor" if "USA Taylor" not in plt.gca().get_legend_handles_labels()[1] else "")
        # Mark launch and target points
        plt.scatter(launch["launch"][1], launch["launch"][0], c="darkgreen", marker="o", s=50, label="USA Launch" if "USA Launch" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(launch["target"][1], launch["target"][0], c="skyblue", marker="x", s=50, label="USA Target" if "USA Target" not in plt.gca().get_legend_handles_labels()[1] else "")
    else:
        plt.plot(f_lons, f_lats, '-', color="orange", label="Russia Fourier" if "Russia Fourier" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot(t_lons, t_lats, '--', color="red", label="Russia Taylor" if "Russia Taylor" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(launch["launch"][1], launch["launch"][0], c="darkorange", marker="o", s=50, label="Russia Launch" if "Russia Launch" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(launch["target"][1], launch["target"][0], c="salmon", marker="x", s=50, label="Russia Target" if "Russia Target" not in plt.gca().get_legend_handles_labels()[1] else "")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Missile Trajectories (Fourier vs. Taylor) with Launch & Target Points")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("missile_trajectories.png")
plt.show()
print("Matplotlib static plot saved as 'missile_trajectories.png'.")

# =============================================================================
# END OF SIMULATION
# =============================================================================
print("\nAll simulations executed. Check the generated HTML and PNG files for results.")