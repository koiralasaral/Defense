import folium
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import TimestampedGeoJson
import random
import math

# -------------------------------
# Define Weapons Data
# -------------------------------
# Each weapon has attributes: name, damage, energy, range, cost
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

# -------------------------------
# Sorting Weapons
# -------------------------------
def sort_weapons(weapons, key):
    return sorted(weapons, key=lambda x: x[key], reverse=True)

# Sort by damage
usa_sorted_damage = sort_weapons(usa_weapons, "damage")
russia_sorted_damage = sort_weapons(russia_weapons, "damage")

# Sort by energy
usa_sorted_energy = sort_weapons(usa_weapons, "energy")
russia_sorted_energy = sort_weapons(russia_weapons, "energy")

# Sort by range
usa_sorted_range = sort_weapons(usa_weapons, "range")
russia_sorted_range = sort_weapons(russia_weapons, "range")

# -------------------------------
# Calculate Weapons Required for Victory
# -------------------------------
damage_threshold = 10000  # Total damage required for victory

def calculate_weapons_required(weapons, threshold):
    results = []
    for weapon in weapons:
        count = math.ceil(threshold / weapon["damage"])
        results.append({"name": weapon["name"], "required": count})
    return results

usa_weapons_required = calculate_weapons_required(usa_sorted_damage, damage_threshold)
russia_weapons_required = calculate_weapons_required(russia_sorted_damage, damage_threshold)

# -------------------------------
# MCMC Simulation
# -------------------------------
def mcmc_simulation(weapons, threshold, iterations=1000):
    results = []
    for weapon in weapons:
        counts = []
        for _ in range(iterations):
            total_damage = 0
            count = 0
            while total_damage < threshold:
                damage = np.random.normal(weapon["damage"], weapon["damage"] * 0.1)  # 10% variability
                total_damage += max(damage, 0)
                count += 1
            counts.append(count)
        avg_count = np.mean(counts)
        results.append({"name": weapon["name"], "avg_required": avg_count})
    return results

usa_mcmc_results = mcmc_simulation(usa_sorted_damage, damage_threshold)
russia_mcmc_results = mcmc_simulation(russia_sorted_damage, damage_threshold)

# -------------------------------
# Simulate War in Folium
# -------------------------------
# Create a Folium map
war_map = folium.Map(location=[50, 0], zoom_start=3)

# Add USA weapons deployment
for idx, weapon in enumerate(usa_mcmc_results):
    folium.Marker(
        location=[38.9072 + random.uniform(-1, 1), -77.0369 + random.uniform(-1, 1)],
        popup=f"{weapon['name']}<br>Avg Required: {weapon['avg_required']:.2f}",
        icon=folium.Icon(color="blue")
    ).add_to(war_map)

# Add Russia weapons deployment
for idx, weapon in enumerate(russia_mcmc_results):
    folium.Marker(
        location=[55.7558 + random.uniform(-1, 1), 37.6173 + random.uniform(-1, 1)],
        popup=f"{weapon['name']}<br>Avg Required: {weapon['avg_required']:.2f}",
        icon=folium.Icon(color="red")
    ).add_to(war_map)

# Save the map
war_map.save("war_simulation.html")
print("War simulation saved as 'war_simulation.html'.")

# -------------------------------
# Visualize Results
# -------------------------------
# Bar chart for USA weapons required
usa_names = [weapon["name"] for weapon in usa_mcmc_results]
usa_avg_required = [weapon["avg_required"] for weapon in usa_mcmc_results]

plt.figure(figsize=(10, 6))
plt.barh(usa_names, usa_avg_required, color="blue")
plt.xlabel("Average Weapons Required")
plt.title("USA Weapons Required for Victory (MCMC Simulation)")
plt.tight_layout()
plt.show()

# Bar chart for Russia weapons required
russia_names = [weapon["name"] for weapon in russia_mcmc_results]
russia_avg_required = [weapon["avg_required"] for weapon in russia_mcmc_results]

plt.figure(figsize=(10, 6))
plt.barh(russia_names, russia_avg_required, color="red")
plt.xlabel("Average Weapons Required")
plt.title("Russia Weapons Required for Victory (MCMC Simulation)")
plt.tight_layout()
plt.show()