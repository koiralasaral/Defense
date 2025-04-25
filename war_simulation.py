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
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlam/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))

# =============================================================================
# INTEGRATED PROGRAM 5 (Time-Stamped Animation for 10 Weapon Systems per Country)
# =============================================================================

# Define 10 weapon systems for USA and Russia with attributes.
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

# Helper function: sort weapons based on a key (here, damage) descending.
def sort_weapons(weapons, key):
    return sorted(weapons, key=lambda x: x[key], reverse=True)

usa_sorted = sort_weapons(usa_weapons, "damage")
russia_sorted = sort_weapons(russia_weapons, "damage")

# Define a victory condition: total damage threshold that must be achieved (for simulation).
damage_threshold = 10000

# Monte Carlo simulation function to estimate number required for each weapon.
def mcmc_simulation(weapons, threshold, iterations=1000):
    results = []
    for weapon in weapons:
        counts = []
        for _ in range(iterations):
            total_damage = 0
            count = 0
            while total_damage < threshold:
                # Simulate damage with 10% variability.
                damage = np.random.normal(weapon["damage"], weapon["damage"] * 0.1)
                total_damage += max(damage, 0)
                count += 1
            counts.append(count)
        avg_count = np.mean(counts)
        results.append({"name": weapon["name"], "avg_required": avg_count})
    return results

usa_mcmc = mcmc_simulation(usa_sorted, damage_threshold)
russia_mcmc = mcmc_simulation(russia_sorted, damage_threshold)

# Print intermediate MCMC results.
print("USA Weapons (MCMC Results):")
for w in usa_mcmc:
    print(f"  {w['name']}: Avg. required = {w['avg_required']:.2f}")
print("Russia Weapons (MCMC Results):")
for w in russia_mcmc:
    print(f"  {w['name']}: Avg. required = {w['avg_required']:.2f}")

# Create a time–stamped animation using Folium for all 10 weapon systems from each country.
war_features = []
base_time = datetime.datetime.now()
# Create features for USA weapons.
for i, weapon in enumerate(usa_mcmc):
    # Slight random perturbation around Washington, DC.
    lat = 38.9072 + random.uniform(-1, 1)
    lon = -77.0369 + random.uniform(-1, 1)
    timestamp = (base_time + datetime.timedelta(minutes=5 * i)).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": timestamp,
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
# Create features for Russia weapons.
for i, weapon in enumerate(russia_mcmc):
    # Slight random perturbation around Moscow.
    lat = 55.7558 + random.uniform(-1, 1)
    lon = 37.6173 + random.uniform(-1, 1)
    timestamp = (base_time + datetime.timedelta(minutes=5 * (i + len(usa_mcmc)))).isoformat()
    feature = {
        "type": "Feature",
        "properties": {
            "time": timestamp,
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
    period="PT5M",  # Each weapon appears 5 minutes apart
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY/MM/DD HH:mm:ss",
    time_slider_drag_update=True
).add_to(war_time_map)
war_time_map.save("integrated_program5_war_simulation_timed.html")
print("Integrated Program 5: War simulation time-stamped animation saved as 'integrated_program5_war_simulation_timed.html'.")

# =============================================================================
# END OF INTEGRATED PROGRAM 5
# =============================================================================
print("\nAll integrated programs executed. Check the generated HTML files and console output for results.")