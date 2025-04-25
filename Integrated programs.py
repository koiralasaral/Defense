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
    Compute a perturbed point along the line from start to end.
    Uses a Fourier (sinusoidal) perturbation added to the linear interpolation.

    Parameters:
      t: parameter in [0,1]
      start, end: (lat, lon) tuples.
      A, B : amplitudes for lat and lon deviations.
      k: frequency multiplier.
      phase: phase shift.

    Returns: (lat, lon) point.
    """
    lat_lin = start[0] + t * (end[0] - start[0])
    lon_lin = start[1] + t * (end[1] - start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)


def taylor_approximation(t0, t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a first–order Taylor series approximation of the Fourier curve about t=t0.
    f(t) ~ f(t0) + f'(t0)*(t-t0)

    Returns approximate (lat, lon) value.
    """
    delta = 1e-5
    p0 = np.array(fourier_curve(t0, start, end, A, B, k, phase))
    p1 = np.array(fourier_curve(t0 + delta, start, end, A, B, k, phase))
    deriv = (p1 - p0) / delta
    return tuple(p0 + deriv * (t - t0))


# =============================================================================
# USER–DEFINED PARAMETERS & DATA
# =============================================================================
# Define capitals:
capitals = {
    "USA": (38.9072, -77.0369),  # Washington, DC
    "Russia": (55.7558, 37.6173)  # Moscow
}
# Defense system specifications:
# USA Patriot: operational_range = 160 km, cost factor = 4
# Russian Pantsir-S1: operational_range = 30 km, cost factor = 3
defense_specs = {
    "USA": {"system": "MIM-104 Patriot", "operational_range": 160, "cost_factor": 4},
    "Russia": {"system": "Pantsir-S1", "operational_range": 30, "cost_factor": 3}
}

# =============================================================================
# INTEGRATED PROGRAM 1 & 3: Deployment Along the Battlefront
# =============================================================================
print("=== Integrated Program 1 & 3: Deployment along Battlefront ===")
# Compute the battlefront using Fourier perturbation.
n_points = 20
t_values = np.linspace(0, 1, n_points)
curve_points = [fourier_curve(t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3) for t in t_values]
taylor_points = [taylor_approximation(0.5, t, capitals["USA"], capitals["Russia"], A=2.0, B=3.0, k=3) for t in t_values]

# At each curve point decide which country covers that point.
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
    required = math.ceil((dist_from_cap / op_range) * cost_fac)
    deployment_data.append({
        "lat": pt[0],
        "lon": pt[1],
        "country": country,
        "required": required
    })

print("Deployment Data:")
for i, d in enumerate(deployment_data):
    print(f"  Point {i}: lat={d['lat']:.4f}, lon={d['lon']:.4f}, country={d['country']}, required={d['required']}")

# --- Create Static Deployment Map (Program 1 part) ---
center_lat = (capitals["USA"][0] + capitals["Russia"][0]) / 2
center_lon = (capitals["USA"][1] + capitals["Russia"][1]) / 2
static_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(static_map)
min_req = min(d["required"] for d in deployment_data)
max_req = max(d["required"] for d in deployment_data)
norm_req = mcolors.Normalize(vmin=min_req, vmax=max_req)
for d in deployment_data:
    norm_val = norm_req(d["required"])
    hex_color = mcolors.to_hex(plt.cm.viridis(norm_val))
    opacity = 0.4 + 0.6 * norm_val
    folium.CircleMarker(
        location=[d["lat"], d["lon"]],
        radius=7,
        popup=f"{d['country']} deployment<br>Required: {d['required']} units",
        color=hex_color,
        fill=True,
        fill_color=hex_color,
        fill_opacity=opacity
    ).add_to(static_map)
static_map.save("integrated_program1_static.html")
print("Integrated Program 1: Static map saved as 'integrated_program1_static.html'.")

# --- Create Time-Stamped Animation Map (Program 3 part) ---
features = []
base_time = datetime.datetime.now()
for i, d in enumerate(deployment_data):
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
geojson = {"type": "FeatureCollection", "features": features}
time_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
TimestampedGeoJson(
    data=geojson,
    period="PT5M",
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(time_map)
folium.PolyLine(locations=curve_points, color="blue", weight=3, opacity=0.7).add_to(time_map)
time_map.save("integrated_program3_timed.html")
print("Integrated Program 3: Timed map saved as 'integrated_program3_timed.html'.")

# =============================================================================
# INTEGRATED PROGRAM 4: MONTE CARLO SIMULATION AT DEPLOYMENT POINTS
# =============================================================================
print("\n=== Integrated Program 4: Monte Carlo Simulation at Deployment Points ===")


def simulate_measurements(true_mu, n=100):
    """Simulate n sensor measurements with 5% noise relative to true_mu."""
    return np.random.normal(true_mu, true_mu * 0.05, size=n)


sim_results = []
for i, d in enumerate(deployment_data):
    sys_mu = 160 if d["country"] == "USA" else 30
    measurements = simulate_measurements(sys_mu, n=100)
    running_mean = np.cumsum(measurements) / np.arange(1, 101)
    final_est = running_mean[-1]
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
    print(f"Point {i}: {d['country']}, Estimated μ={final_est:.2f}, MC required={req_est}")

# Bar chart for Monte Carlo results.
indices = np.arange(len(sim_results))
mc_counts = [r["mc_required"] for r in sim_results]
plt.figure(figsize=(10, 6))
plt.bar(indices, mc_counts, color="orchid")
plt.xlabel("Deployment Point Index")
plt.ylabel("Estimated Weapons Required (MC Simulation)")
plt.title("Monte Carlo Simulation at Deployment Points")
plt.xticks(indices, [str(r["point_index"]) for r in sim_results])
plt.tight_layout()
plt.savefig("integrated_program4_mc_simulation.png")
plt.show()
print("Integrated Program 4: MC simulation bar chart saved as 'integrated_program4_mc_simulation.png'.")

# Also create a Folium choropleth map for estimated μ.
mc_map = folium.Map(location=[center_lat, center_lon], zoom_start=4)
all_est = [r["final_est_mu"] for r in sim_results]
est_norm = mcolors.Normalize(vmin=min(all_est), vmax=max(all_est))
for idx, r in enumerate(sim_results):
    col = mcolors.to_hex(plt.cm.plasma(est_norm(r["final_est_mu"])))
    folium.CircleMarker(
        location=[deployment_data[idx]["lat"], deployment_data[idx]["lon"]],
        radius=6,
        color=col,
        fill=True,
        fill_color=col,
        fill_opacity=0.7,
        popup=(f"{deployment_data[idx]['country']} system,<br>True μ = "
               f"{160 if deployment_data[idx]['country'] == 'USA' else 30} km,<br>"
               f"Estimated μ = {r['final_est_mu']:.2f} km")
    ).add_to(mc_map)
mc_map.save("integrated_program4_mc_choropleth.html")
print("Integrated Program 4: MC choropleth map saved as 'integrated_program4_mc_choropleth.html'.")

# =============================================================================
# INTEGRATED PROGRAM 5 (Converted): WAR SIMULATION Time-Stamped Animation for 10 Weapons per Country
# =============================================================================
print("\n=== Integrated Program 5: War Simulation Time-Stamped Animation ===")
# Define 10 weapon systems per country:
usa_weapons = [
    {"name": "MIM-104 Patriot", "damage": 800, "energy": 300, "range": 160, "cost": 4000000},
    {"name": "THAAD", "damage": 1200, "energy": 350, "range": 200, "cost": 10000000},
    {"name": "Avenger", "damage": 600, "energy": 250, "range": 140, "cost": 2000000},
    {"name": "NASAMS", "damage": 900, "energy": 280, "range": 170, "cost": 5000000},
    {"name": "Aegis BMD", "damage": 1500, "energy": 400, "range": 220, "cost": 12000000},
    {"name": "Iron Dome", "damage": 700, "energy": 260, "range": 150, "cost": 3000000},
    {"name": "Patriot PAC-3", "damage": 1000, "energy": 320, "range": 180, "cost": 7000000},
    {"name": "Sentinel Radar", "damage": 500, "energy": 200, "range": 120, "cost": 1500000},
    {"name": "Arrow 3", "damage": 1300, "energy": 370, "range": 210, "cost": 11000000},
    {"name": "SM-3", "damage": 1400, "energy": 390, "range": 230, "cost": 13000000}
]
russia_weapons = [
    {"name": "Pantsir-S1", "damage": 700, "energy": 260, "range": 30, "cost": 3000000},
    {"name": "S-400", "damage": 1100, "energy": 330, "range": 180, "cost": 8000000},
    {"name": "Buk-M2", "damage": 850, "energy": 270, "range": 100, "cost": 4000000},
    {"name": "Tor-M2", "damage": 750, "energy": 240, "range": 90, "cost": 2500000},
    {"name": "S-300", "damage": 1000, "energy": 320, "range": 190, "cost": 7000000},
    {"name": "Iskander-M", "damage": 1500, "energy": 400, "range": 500, "cost": 12000000},
    {"name": "Kinzhal", "damage": 2000, "energy": 450, "range": 2000, "cost": 20000000},
    {"name": "Kalibr", "damage": 1800, "energy": 420, "range": 1500, "cost": 18000000},
    {"name": "Bastion-P", "damage": 1300, "energy": 370, "range": 300, "cost": 11000000},
    {"name": "Sarmat", "damage": 2500, "energy": 500, "range": 18000, "cost": 50000000}
]


def sort_weapons(weapons, key):
    return sorted(weapons, key=lambda x: x[key], reverse=True)


# Sort by damage as an example.
usa_sorted_damage = sort_weapons(usa_weapons, "damage")
russia_sorted_damage = sort_weapons(russia_weapons, "damage")

# Define victory condition: total damage threshold.
damage_threshold = 10000


def mcmc_simulation(weapons, threshold, iterations=1000):
    results = []
    for weapon in weapons:
        counts = []
        for _ in range(iterations):
            total_damage = 0
            count = 0
            while total_damage < threshold:
                damage = np.random.normal(weapon["damage"], weapon["damage"] * 0.1)  # 10% noise
                total_damage += max(damage, 0)
                count += 1
            counts.append(count)
        avg_count = np.mean(counts)
        results.append({"name": weapon["name"], "avg_required": avg_count})
    return results


usa_mcmc = mcmc_simulation(usa_sorted_damage, damage_threshold)
russia_mcmc = mcmc_simulation(russia_sorted_damage, damage_threshold)

print("\nUSA Weapons (MCMC Results):")
for w in usa_mcmc:
    print(f"  {w['name']}: Avg. required = {w['avg_required']:.2f}")
print("\nRussia Weapons (MCMC Results):")
for w in russia_mcmc:
    print(f"  {w['name']}: Avg. required = {w['avg_required']:.2f}")

# Convert the war simulation into a time-stamped animated deployment map.
war_features = []
base_time = datetime.datetime.now()
# For USA weapons near Washington, DC.
for i, weapon in enumerate(usa_mcmc):
    lat = 38.9072 + random.uniform(-1, 1)
    lon = -77.0369 + random.uniform(-1, 1)
    tstamp = (base_time + datetime.timedelta(minutes=10 * i)).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": tstamp,
            "popup": f"USA: {weapon['name']}<br>Avg Required: {weapon['avg_required']:.2f}",
            "icon": "circle",
            "iconstyle": {
                "fillColor": "blue",
                "fillOpacity": 0.8,
                "stroke": "true",
                "radius": 7
            }
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        }
    }
    war_features.append(feature)
# For Russia weapons near Moscow.
for i, weapon in enumerate(russia_mcmc):
    lat = 55.7558 + random.uniform(-1, 1)
    lon = 37.6173 + random.uniform(-1, 1)
    tstamp = (base_time + datetime.timedelta(minutes=10 * (i + len(usa_mcmc)))).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": tstamp,
            "popup": f"Russia: {weapon['name']}<br>Avg Required: {weapon['avg_required']:.2f}",
            "icon": "circle",
            "iconstyle": {
                "fillColor": "red",
                "fillOpacity": 0.8,
                "stroke": "true",
                "radius": 7
            }
        },
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        }
    }
    war_features.append(feature)

war_geojson = {"type": "FeatureCollection", "features": war_features}
war_time_map = folium.Map(location=[50, 0], zoom_start=3)
TimestampedGeoJson(
    data=war_geojson,
    period="PT10M",  # Each time step = 10 minutes
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(war_time_map)
war_time_map.save("integrated_program5_war_simulation_timed.html")
print(
    "Integrated Program 5: War simulation time-stamped animation saved as 'integrated_program5_war_simulation_timed.html'.")

# =============================================================================
# END OF ALL INTEGRATED PROGRAMS
# =============================================================================
print("\nAll integrated programs executed. Check the generated HTML and PNG files for results.")