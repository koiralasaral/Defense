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
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
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
    Compute a first–order Taylor series (linear) approximation of our Fourier curve about t=t0.
    First, compute the point at t0 and a numerical derivative. Then approximate:
      f(t) ~ f(t0) + f'(t0) * (t - t0)
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0 + delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return tuple(p0 + deriv * (t - t0))

# =============================================================================
# USER–DEFINED PARAMETERS & DATA
# =============================================================================
# Define capitals (for our two countries)
capitals = {
    "USA": (38.9072, -77.0369),   # Washington, DC
    "Russia": (55.7558, 37.6173)   # Moscow
}

# For our two defense systems:
# US Patriot: operational_range (km) = 160, cost factor = 4 (arbitrary unit)
# Russian Pantsir-S1: operational_range (km) = 30, cost factor = 3
defense_specs = {
    "USA": {"system": "MIM-104 Patriot", "operational_range": 160, "cost_factor": 4},
    "Russia": {"system": "Pantsir-S1", "operational_range": 30, "cost_factor": 3}
}

# =============================================================================
# PROGRAM 1: FOLIUM STATIC MAP WITH CHOROPLETH (DEPLOYMENT ALONG THE SEPARATION CURVE)
# =============================================================================
print("=== PROGRAM 1: FOLIUM STATIC MAP WITH CHOROPLETH ===")
# (a) Compute the separation (battlefront) curve
n_points = 20
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3, phase=0) for t in t_values]

# (b) For extra analysis, compute a Taylor series approximation at the mid–point t=0.5.
taylor_points = []
for t in t_values:
    approx = taylor_approximation(0.5, t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3, phase=0)
    taylor_points.append(tuple(approx))

# (c) At each curve point, decide which country will cover that point (by proximity)
deployment_data = []
for pt in curve_points:
    d_US = haversine(capitals["USA"][0], capitals["USA"][1], pt[0], pt[1])
    d_Rus = haversine(capitals["Russia"][0], capitals["Russia"][1], pt[0], pt[1])
    if d_US < d_Rus:
        country = "USA"
        op_range = defense_specs["USA"]["operational_range"]
        cost_fac = defense_specs["USA"]["cost_factor"]
        dist_from_cap = d_US
    else:
        country = "Russia"
        op_range = defense_specs["Russia"]["operational_range"]
        cost_fac = defense_specs["Russia"]["cost_factor"]
        dist_from_cap = d_Rus
    # Compute required weapons at this point (a simple formula)
    required = math.ceil((dist_from_cap / op_range) * cost_fac)
    deployment_data.append({
        "lat": pt[0],
        "lon": pt[1],
        "country": country,
        "required": required
    })

# Print intermediate deployment data.
for idx, d in enumerate(deployment_data):
    print(f"Point {idx}: lat={d['lat']:.4f}, lon={d['lon']:.4f}, country={d['country']}, required={d['required']}")

# (d) Create a Folium map showing the battlefront curve and deployment markers.
center_lat = (capitals["USA"][0] + capitals["Russia"][0]) / 2
center_lon = (capitals["USA"][1] + capitals["Russia"][1]) / 2
folium_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
# Draw the battlefront curve
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(folium_map)

# Setup a colormap (viridis) for choropleth style based on "required" value.
min_req = min(d["required"] for d in deployment_data)
max_req = max(d["required"] for d in deployment_data)
norm_req = mcolors.Normalize(vmin=min_req, vmax=max_req)
for d in deployment_data:
    norm_val = norm_req(d["required"])
    hex_color = mcolors.to_hex(plt.cm.viridis(norm_val))
    opacity = 0.4 + 0.6 * norm_val  # opacity scales from 0.4 to 1.0
    folium.CircleMarker(
        location=[d["lat"], d["lon"]],
        radius=7,
        popup=f"{d['country']} deployment<br>Required: {d['required']} units",
        color=hex_color,
        fill=True,
        fill_color=hex_color,
        fill_opacity=opacity
    ).add_to(folium_map)

folium_map.save("program1_deployment_static.html")
print("PROGRAM 1: Folium static choropleth map saved as 'program1_deployment_static.html'.")

# =============================================================================
# PROGRAM 2: MATPLOTLIB STATIC VISUALIZATION (CURVE & TAYLOR APPROXIMATION)
# =============================================================================
print("\n=== PROGRAM 2: MATPLOTLIB STATIC VISUALIZATION ===")
plt.figure(figsize=(10,6))
# Plot Fourier–generated battlefront curve.
curve_lats = [pt[0] for pt in curve_points]
curve_lons = [pt[1] for pt in curve_points]
plt.plot(curve_lons, curve_lats, 'b.-', label="Fourier Curve")
# Plot Taylor approximation curve.
taylor_lats = [pt[0] for pt in taylor_points]
taylor_lons = [pt[1] for pt in taylor_points]
plt.plot(taylor_lons, taylor_lats, 'r--', label="Taylor Approx. (t0=0.5)")
# Mark deployment points with text showing required units.
for d in deployment_data:
    plt.plot(d["lon"], d["lat"], 'ko')
    plt.text(d["lon"], d["lat"], f" {d['required']}", color="purple", fontsize=8)
plt.title("Battlefront Curve & Taylor Approximation with Deployment Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("program2_fourier_taylor_static.png")
plt.show()
print("PROGRAM 2: Matplotlib static visualization saved as 'program2_fourier_taylor_static.png'.")

# =============================================================================
# PROGRAM 3: TIME-STAMPED FOLIUM ANIMATION (DEPLOYMENT ALONG THE CURVE)
# =============================================================================
print("\n=== PROGRAM 3: TIME-STAMPED FOLIUM ANIMATION ===")
features = []
base_time = datetime.datetime.now()
for i, d in enumerate(deployment_data):
    # Increase timestamp by 5 minutes per deployment point.
    tstamp = (base_time + datetime.timedelta(minutes=5 * i)).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": tstamp,
            "popup": f"{d['country']} deployment<br>Required: {d['required']}",
            "icon": "circle",
            "iconstyle": {
                "fillColor": mcolors.to_hex(plt.cm.viridis(norm_req(d["required"]))),
                "fillOpacity": 0.4 + 0.6 * norm_req(d["required"]),
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
folium_anim_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
TimestampedGeoJson(
    data=geojson,
    period="PT5M",  # each time step is 5 minutes
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(folium_anim_map)
# Also add the battlefront polyline for context.
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(folium_anim_map)
folium_anim_map.save("program3_deployment_timed.html")
print("PROGRAM 3: Time-stamped Folium animation saved as 'program3_deployment_timed.html'.")

# =============================================================================
# PROGRAM 4: MONTE CARLO SIMULATION & EFFICIENT ESTIMATOR (COST EFFICIENCY)
# =============================================================================
print("\n=== PROGRAM 4: MONTE CARLO SIMULATION & EFFICIENT ESTIMATOR ===")
def simulate_measurements(true_mu, n=100):
    """Simulate n sensor measurements with 5% noise relative to true_mu."""
    return np.random.normal(true_mu, true_mu * 0.05, size=n)

sim_results = []
for i, d in enumerate(deployment_data):
    # Decide local true operational range based on system affiliation.
    sys_mu = 160 if d["country"] == "USA" else 30
    measurements = simulate_measurements(sys_mu, n=100)
    running_mean = np.cumsum(measurements) / np.arange(1, 101)
    final_est = running_mean[-1]
    # Recompute required weapons using the efficient estimator (final_est) and same formula.
    if d["country"] == "USA":
        req_est = math.ceil((haversine(capitals["USA"][0], capitals["USA"][1], d["lat"], d["lon"]) / final_est)
                            * defense_specs["USA"]["cost_factor"])
    else:
        req_est = math.ceil((haversine(capitals["Russia"][0], capitals["Russia"][1], d["lat"], d["lon"]) / final_est)
                            * defense_specs["Russia"]["cost_factor"])
    sim_results.append({
        "point_index": i,
        "country": d["country"],
        "final_est_mu": final_est,
        "mc_required": req_est
    })
    print(f"Deployment point {i}: country={d['country']}, final_estimated μ={final_est:.2f}, MC required={req_est}")

# Plot a bar chart of Monte Carlo–based required count.
indices = np.arange(len(sim_results))
mc_counts = [r["mc_required"] for r in sim_results]
plt.figure(figsize=(10,6))
plt.bar(indices, mc_counts, color="orchid")
plt.xlabel("Deployment Point Index (along battlefront)")
plt.ylabel("Estimated Weapons Required (MC Simulation)")
plt.title("Monte Carlo Simulation for Cost-Efficient Weapon Requirement")
plt.xticks(indices, [str(r["point_index"]) for r in sim_results])
plt.tight_layout()
plt.savefig("program4_mc_simulation.png")
plt.show()
print("PROGRAM 4: Monte Carlo simulation bar chart saved as 'program4_mc_simulation.png'.")

# Create a Folium choropleth based on estimated μ.
mc_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
all_est = [r["final_est_mu"] for r in sim_results]
est_norm = mcolors.Normalize(vmin=min(all_est), vmax=max(all_est))
for dr, idx in zip(sim_results, range(len(sim_results))):
    col = mcolors.to_hex(plt.cm.plasma(est_norm(dr["final_est_mu"])))
    folium.CircleMarker(
        location=[deployment_data[idx]["lat"], deployment_data[idx]["lon"]],
        radius=6,
        color=col,
        fill=True,
        fill_color=col,
        fill_opacity=0.7,
        popup=(f"{deployment_data[idx]['country']} system, True μ = "
               f"{160 if deployment_data[idx]['country']=='USA' else 30} km, Estimated μ = {dr['final_est_mu']:.2f} km")
    ).add_to(mc_map)
mc_map.save("program4_mc_choropleth.html")
print("PROGRAM 4: MC choropleth map saved as 'program4_mc_choropleth.html'.")

print("\nAll programs executed. Check the generated HTML and PNG files for results.")