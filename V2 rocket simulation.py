import folium
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from folium.plugins import TimestampedGeoJson
import math, random, json, datetime

# =============================================================================
# COMMON FUNCTIONS
# =============================================================================

def haversine(lat1, lon1, lat2, lon2):
    """Compute the great–circle distance (in km) between two (lat,lon) points."""
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a perturbed point along the line from start to end.
    Uses a Fourier (sinusoidal) perturbation added to the straight‐line interpolation.
      t: parameter between 0 and 1.
      start, end: (lat,lon) tuples.
      A, B: amplitudes for lat and lon deviations.
    """
    lat_lin = start[0] + t*(end[0] - start[0])
    lon_lin = start[1] + t*(end[1] - start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)

def taylor_approximation(t0, t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a first–order Taylor series (linear) approximation of our Fourier curve
    about t=t0.
    First, compute the point at t0 and a numerical derivative.
    Then approximate:
      f(t) ~ f(t0) + f'(t0) * (t - t0)
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0 + delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return p0 + deriv * (t - t0)

# =============================================================================
# USER–DEFINED PARAMETERS & DATA
# =============================================================================

# Define capitals (for our two countries)
capitals = {
    "UK": (51.5074, 0.1278),   # London, UK
    "Germany": (52.5200, 13.4050)   # Berlin, Germany
}

# V2 rocket parameters:
v2_operational_range = 320  # km
v2_cost_factor = 1  # Arbitrary unit

# For our two defense systems  (now both the same, effectively):
defense_specs = {
    "UK": {"system": "V2 Rocket", "operational_range": v2_operational_range, "cost_factor": v2_cost_factor},
    "Germany": {"system": "V2 Rocket", "operational_range": v2_operational_range, "cost_factor": v2_cost_factor}
}

# =============================================================================
# PROGRAM 1: FOLIUM STATIC MAP WITH CHOROPLETH (DEPLOYMENT ALONG THE SEPARATION CURVE)
# =============================================================================

# (a) Compute the separation (battlefront) curve
n_points = 20
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["UK"], capitals["Germany"], A=2.0, B=3.0, k=3, phase=0) for t in t_values]

# (b) For extra analysis, compute a Taylor series approximation at the mid-point t=0.5
taylor_points = []
for t in t_values:
    taylor_points.append(taylor_approximation(0.5, t, capitals["UK"], capitals["Germany"], A=2.0, B=3.0, k=3, phase=0))

# (c) At each curve point, decide which country will cover that point (by proximity)
deployment_data = []
for pt in curve_points:
    d_UK = haversine(capitals["UK"][0], capitals["UK"][1], pt[0], pt[1])
    d_Germany = haversine(capitals["Germany"][0], capitals["Germany"][1], pt[0], pt[1])
    if d_UK < d_Germany:
        country = "UK"
        op_range = defense_specs["UK"]["operational_range"]
        cost_fac = defense_specs["UK"]["cost_factor"]
        dist_from_cap = d_UK
    else:
        country = "Germany"
        op_range = defense_specs["Germany"]["operational_range"]
        cost_fac = defense_specs["Germany"]["cost_factor"]
        dist_from_cap = d_Germany
    # Compute required weapons at this point (simple formula):
    required = math.ceil((dist_from_cap / op_range) * cost_fac)
    deployment_data.append({
        "lat": pt[0],
        "lon": pt[1],
        "country": country,
        "required": required
    })

# (d) Create a Folium map showing the battlefront curve and deployment markers.
# First, create a base map centered between capitals.
center_lat = (capitals["UK"][0] + capitals["Germany"][0]) / 2
center_lon = (capitals["UK"][1] + capitals["Germany"][1]) / 2
folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)

# Draw the battlefront curve (as a polyline)
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(folium_map)

# Find minimum and maximum required count to normalize color and opacity later.
min_req = min(d["required"] for d in deployment_data)
max_req = max(d["required"] for d in deployment_data)

# Setup a Matplotlib colormap for the choropleth style.
norm = mcolors.Normalize(vmin=min_req, vmax=max_req)
cmap = plt.cm.viridis

for data in deployment_data:
    # normalized value for color and opacity (opacity scaled between 0.4 and 1.0)
    norm_val = norm(data["required"])
    hex_color = mcolors.to_hex(cmap(norm_val))
    opacity = 0.4 + 0.6 * norm_val  # from 0.4 to 1.0
    folium.CircleMarker(
        location=[data["lat"], data["lon"]],
        radius=7,
        popup=f"{data['country']} deployment<br>Required: {data['required']} units",
        color=hex_color,
        fill=True,
        fill_color=hex_color,
        fill_opacity=opacity
    ).add_to(folium_map)

# Save the static choropleth map.
folium_map.save("program1_v2_deployment_static.html")
print("PROGRAM 1: Folium static choropleth map saved as 'program1_v2_deployment_static.html'.")

# =============================================================================
# PROGRAM 2: MATPLOTLIB STATIC VISUALIZATION (CURVE & TAYLOR APPROXIMATION)
# =============================================================================

plt.figure(figsize=(10, 6))
# Plot the Fourier–generated battlefront curve.
curve_lats = [pt[0] for pt in curve_points]
curve_lons = [pt[1] for pt in curve_points]
plt.plot(curve_lons, curve_lats, 'b.-', label="Fourier Curve")

# Plot the Taylor series approximation curve.
taylor_lats = [pt[0] for pt in taylor_points]
taylor_lons = [pt[1] for pt in taylor_points]
plt.plot(taylor_lons, taylor_lats, 'r--', label="Taylor Approx. (around t=0.5)")

# Mark the deployment points.
for d in deployment_data:
    plt.plot(d["lon"], d["lat"], 'ko')
    plt.text(d["lon"], d["lat"], f" {d['required']}", color="purple", fontsize=8)

plt.title("V2 Rocket Trajectory and Taylor Approximation with Deployment Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================================================================
# PROGRAM 3: TIME-STAMPED FOLIUM ANIMATION (DEPLOYMENT ALONG THE CURVE)
# =============================================================================

# For animation we create a GeoJSON FeatureCollection.
# Each deployment point will have a timestamp (simulate by using an increasing time string).
features = []
start_time = datetime.datetime.now()
for i, d in enumerate(deployment_data):
    # Increase timestamp by, say, 5 minutes for each successive point.
    tstamp = (start_time + datetime.timedelta(minutes=5*i)).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": tstamp,
            "popup": f"{d['country']} deployment<br>Required: {d['required']}",
            "icon": "circle",
            "iconstyle": {
                "fillColor": mcolors.to_hex(cmap(norm(d["required"]))),
                "fillOpacity": 0.4 + 0.6 * norm(d["required"]),
                "stroke": "true",
                "radius": 7
            }
        },
        "geometry": {
            "type": "Point",
            "coordinates": [d["lon"], d["lat"]]
        }
    }
    features.append(feature)

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Create a folium map and add the TimestampedGeoJson layer.
folium_anim_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
TimestampedGeoJson(
    geojson,
    period="PT5M",  # period of 5 minutes
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options='YYYY/MM/DD HH:mm:ss',
    time_slider_drag_update=True
).add_to(folium_anim_map)

# Also add the battlefront polyline.
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(folium_anim_map)

folium_anim_map.save("program3_v2_deployment_timed.html")
print("PROGRAM 3: Time-stamped Folium animation saved as 'program3_v2_deployment_timed.html'.")

# =============================================================================
# PROGRAM 4: MONTE CARLO SIMULATION & EFFICIENT ESTIMATOR (COST EFFICIENCY)
# =============================================================================

# We simulate sensor measurements at each deployment point to estimate the local operational range μ.
# Assume the measurement noise is 5% of the true value.
def simulate_measurements(true_mu, n=100):
    return np.random.normal(true_mu, true_mu*0.05, size=n)

# For each deployment point, choose the "true" μ based on the system in use.
# For UK & Germany deployment, true μ = v2_operational_range.
sim_results = []
for i, d in enumerate(deployment_data):
    sys_mu = v2_operational_range
    measurements = simulate_measurements(sys_mu, n=100)
    running_mu = np.cumsum(measurements) / np.arange(1, 101)
    final_est = running_mu[-1]
    # Recompute required weapons using the efficient estimator (final_est) and same formula.
    if d["country"] == "UK":
        req_est = math.ceil((haversine(capitals["UK"][0], capitals["UK"][1], d["lat"], d["lon"]) / final_est) * defense_specs["UK"]["cost_factor"])
    else:
        req_est = math.ceil((haversine(capitals["Germany"][0], capitals["Germany"][1], d["lat"], d["lon"]) / final_est) * defense_specs["Germany"]["cost_factor"])
    sim_results.append({
        "point_index": i,
        "country": d["country"],
        "final_est_mu": final_est,
        "mc_required": req_est
    })

# Plot a bar chart comparing the Monte Carlo–based required count for each deployment point.
plt.figure(figsize=(10, 6))
indices = np.arange(len(sim_results))
mc_counts = [r["mc_required"] for r in sim_results]
plt.bar(indices, mc_counts, color="orchid")
plt.xlabel("Deployment Point Index (along battlefront)")
plt.ylabel("Estimated Weapons Required (MC Simulation)")
plt.title("Monte Carlo Simulation of V2 Rocket Requirement")
plt.xticks(indices, [str(r["point_index"]) for r in sim_results])
plt.tight_layout()
plt.show()

# =============================================================================
# END OF PROGRAMS
# =============================================================================

print("\nAll programs executed. Check the generated HTML files and plots for results.")
