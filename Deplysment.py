import folium
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import math
import random

# =============================================================================
# Program 1: Border Calculation, Separation Line, and Victory Weapon Count
# =============================================================================

# Define capitals coordinates (Washington, DC & Moscow)
capitals = {
    "Washington, DC": (38.9072, -77.0369),
    "Moscow": (55.7558, 37.6173)
}

# Haversine formula (in km)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    distance = 2 * R * np.arcsin(np.sqrt(a))
    return distance

# Compute distance between capitals
d_total = haversine(capitals["Washington, DC"][0], capitals["Washington, DC"][1],
                     capitals["Moscow"][0], capitals["Moscow"][1])
print("Distance between capitals: {:.2f} km".format(d_total))

# Based on system performance:
# US Patriot: operational range = 160 km
# Russian Pantsir-S1: operational range = 30 km
# A performance–based dividing line is placed at: x = Patriot_range / (Patriot_range + Pantsir_range)
ratio = 160 / (160 + 30)  # ~0.8421 from US side
print("Separation ratio from US side: {:.4f}".format(ratio))

# Compute the separation (battlefront) point along the straight line joining capitals
lat_sep = capitals["Washington, DC"][0] + ratio * (capitals["Moscow"][0] - capitals["Washington, DC"][0])
lon_sep = capitals["Washington, DC"][1] + ratio * (capitals["Moscow"][1] - capitals["Washington, DC"][1])
separation_point = (lat_sep, lon_sep)

# Compute distances from capitals to the separation point
d_US = haversine(capitals["Washington, DC"][0], capitals["Washington, DC"][1], lat_sep, lon_sep)
d_Russia = haversine(lat_sep, lon_sep, capitals["Moscow"][0], capitals["Moscow"][1])
print("Distance from Washington, DC to separation: {:.2f} km".format(d_US))
print("Distance from separation to Moscow: {:.2f} km".format(d_Russia))

# Estimate the number of weapons required to cover the distance
# (Assuming each defense weapon “covers” a distance equal to its operational range.)
num_US = math.ceil(d_US / 160)
num_Russia = math.ceil(d_Russia / 30)
print("Weapons needed from US side (Patriot):", num_US)
print("Weapons needed from Russian side (Pantsir-S1):", num_Russia)

# --- Matplotlib Visualization ---
plt.figure(figsize=(8,6))
plt.plot([capitals["Washington, DC"][1], capitals["Moscow"][1]],
         [capitals["Washington, DC"][0], capitals["Moscow"][0]], 'k-o', label="Capital Connection")
plt.plot(lon_sep, lat_sep, 'ro', markersize=8, label="Separation Point")
plt.text(capitals["Washington, DC"][1], capitals["Washington, DC"][0], " Washington, DC", fontsize=9)
plt.text(capitals["Moscow"][1], capitals["Moscow"][0], " Moscow", fontsize=9)
plt.text(lon_sep, lat_sep, " Separation", fontsize=9)
plt.title("Separation Line between Capitals\nTotal distance: {:.2f} km".format(d_total))
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()

# --- Folium Map ---
m1 = folium.Map(
    location=[(capitals["Washington, DC"][0] + capitals["Moscow"][0]) / 2,
              (capitals["Washington, DC"][1] + capitals["Moscow"][1]) / 2],
    zoom_start=3
)
# Add capital markers
for city, coords in capitals.items():
    folium.Marker(location=coords, popup=city).add_to(m1)
# Add separation point
folium.Marker(
    location=separation_point,
    popup="Separation Point",
    icon=folium.Icon(color="red")
).add_to(m1)
# Draw polyline connecting capitals via the separation point
folium.PolyLine(
    locations=[capitals["Washington, DC"], separation_point, capitals["Moscow"]],
    color="blue", weight=2.5, opacity=1
).add_to(m1)
m1.save("border_separation_map.html")
print("Border separation map saved as 'border_separation_map.html'.")

# =============================================================================
# Program 2: Monte Carlo Simulation for 10 Weapons and Folium Usage
# =============================================================================

# Define 10 example weapons (5 USA, 5 Russia)
weapons = [
    {"name": "MIM-104 Patriot", "country": "USA", "weight": 5000, "cost": 4000000,
     "destruction": 800, "power_required": 100, "energy_output": 300, "efficiency": 0.85,
     "performance_mu": 160},
    {"name": "THAAD", "country": "USA", "weight": 20000, "cost": 10000000,
     "destruction": 1200, "power_required": 150, "energy_output": 350, "efficiency": 0.88,
     "performance_mu": 200},
    {"name": "Avenger", "country": "USA", "weight": 3000, "cost": 2000000,
     "destruction": 600, "power_required": 80, "energy_output": 250, "efficiency": 0.80,
     "performance_mu": 140},
    {"name": "NASAMS", "country": "USA", "weight": 7000, "cost": 5000000,
     "destruction": 900, "power_required": 110, "energy_output": 280, "efficiency": 0.86,
     "performance_mu": 170},
    {"name": "Aegis BMD", "country": "USA", "weight": 25000, "cost": 12000000,
     "destruction": 1500, "power_required": 200, "energy_output": 400, "efficiency": 0.90,
     "performance_mu": 220},
    {"name": "Pantsir-S1", "country": "Russia", "weight": 6000, "cost": 3000000,
     "destruction": 700, "power_required": 90, "energy_output": 260, "efficiency": 0.82,
     "performance_mu": 30},
    {"name": "S-400", "country": "Russia", "weight": 15000, "cost": 8000000,
     "destruction": 1100, "power_required": 130, "energy_output": 330, "efficiency": 0.87,
     "performance_mu": 180},
    {"name": "Buk-M2", "country": "Russia", "weight": 8000, "cost": 4000000,
     "destruction": 850, "power_required": 95, "energy_output": 270, "efficiency": 0.83,
     "performance_mu": 100},
    {"name": "Tor-M2", "country": "Russia", "weight": 4000, "cost": 2500000,
     "destruction": 750, "power_required": 85, "energy_output": 240, "efficiency": 0.81,
     "performance_mu": 90},
    {"name": "S-300", "country": "Russia", "weight": 13000, "cost": 7000000,
     "destruction": 1000, "power_required": 120, "energy_output": 320, "efficiency": 0.86,
     "performance_mu": 190}
]

# For Monte Carlo simulation, assume victory is reached when cumulative damage exceeds a threshold.
damage_threshold = 10000
n_simulations = 10000

def simulate_weapons_required(weapon):
    required_counts = []
    mean_destruction = weapon["destruction"]
    sd_destruction = mean_destruction * 0.1  # assume 10% variability
    for _ in range(n_simulations):
        total_damage = 0
        count = 0
        while total_damage < damage_threshold:
            damage = np.random.normal(mean_destruction, sd_destruction)
            total_damage += max(damage, 0)
            count += 1
        required_counts.append(count)
    return np.mean(required_counts), np.std(required_counts)

weapon_requirements = {}
for weapon in weapons:
    avg_required, std_required = simulate_weapons_required(weapon)
    weapon_requirements[weapon["name"]] = (avg_required, std_required)
    print(f"{weapon['name']} ({weapon['country']}): Avg. required = {avg_required:.2f} (std = {std_required:.2f})")

# --- Folium Map for Weapons ---
capitals_details = {
    "USA": {"capital": (38.9072, -77.0369)},
    "Russia": {"capital": (55.7558, 37.6173)}
}
m2 = folium.Map(location=[45, 0], zoom_start=3)
popup1 = folium.LatLngPopup()

m2.add_child(popup1)
# Add capital markers
for country, info in capitals_details.items():
    folium.Marker(info["capital"],
                  popup=country,
                  icon=folium.Icon(color="green" if country=="USA" else "blue")).add_to(m2)
# Add markers for each weapon near its country's capital (with slight random offset)
for weapon in weapons:
    country = weapon["country"]
    base_lat, base_lon = capitals_details["USA"]["capital"] if country=="USA" else capitals_details["Russia"]["capital"]
    pert_lat = base_lat + random.uniform(-1, 1)
    pert_lon = base_lon + random.uniform(-1, 1)
    req = weapon_requirements[weapon["name"]][0]
    folium.CircleMarker(
        location=[pert_lat, pert_lon],
        radius=5,
        popup=f"{weapon['name']}: ~{req:.1f} units needed",
        color="orange",
        fill=True,
        fill_color="orange",
        fill_opacity=0.8
    ).add_to(m2)
m2.save("weapons_victory_map.html")

print("Weapons victory map saved as 'weapons_victory_map.html'.")

# --- Matplotlib Animation for one weapon (MIM-104 Patriot) ---
patriot = [w for w in weapons if w["name"] == "MIM-104 Patriot"][0]
counts = []
running_avg = []
cum_sum = 0
for sim in range(1, n_simulations + 1):
    total_damage = 0
    count = 0
    while total_damage < damage_threshold:
        damage = np.random.normal(patriot["destruction"], patriot["destruction"] * 0.1)
        total_damage += max(damage, 0)
        count += 1
    counts.append(count)
    cum_sum += count
    running_avg.append(cum_sum / sim)

fig2, ax2 = plt.subplots()
line, = ax2.plot([], [], 'bo-', lw=2)
ax2.set_xlim(0, n_simulations)
ax2.set_ylim(min(counts) - 1, max(counts) + 1)
ax2.set_xlabel("Simulation")
ax2.set_ylabel("Running Avg. Required Count")
ax2.set_title("Monte Carlo for MIM-104 Patriot")
def update_mc(frame):
    xdata = np.arange(1, frame + 1)
    ydata = running_avg[:frame]
    line.set_data(xdata, ydata)
    return line,
# We step through a subset of frames for speed
anim_mc = FuncAnimation(fig2, update_mc, frames=range(1, n_simulations, max(1, n_simulations//100)),
                        interval=50, blit=True, repeat=False)
plt.show()

# =============================================================================
# Program 3: Sorting Weapons by Weight (Descending)
# =============================================================================
sorted_by_weight = sorted(weapons, key=lambda x: x["weight"], reverse=True)
print("\nWeapons sorted by Weight (Descending):")
for w in sorted_by_weight:
    print(f"{w['name']} ({w['country']}): {w['weight']} kg")
plt.figure(figsize=(10,6))
names = [w["name"] for w in sorted_by_weight]
weights = [w["weight"] for w in sorted_by_weight]
plt.bar(names, weights, color='skyblue')
plt.title("Weapons Sorted by Weight (Descending)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Weight (kg)")
plt.tight_layout()
plt.show()

# =============================================================================
# Program 4: Sorting Weapons by Cost (Descending)
# =============================================================================
sorted_by_cost = sorted(weapons, key=lambda x: x["cost"], reverse=True)
print("\nWeapons sorted by Cost (Descending):")
for w in sorted_by_cost:
    print(f"{w['name']} ({w['country']}): ${w['cost']}")
plt.figure(figsize=(10,6))
names = [w["name"] for w in sorted_by_cost]
costs = [w["cost"] for w in sorted_by_cost]
plt.bar(names, costs, color='salmon')
plt.title("Weapons Sorted by Cost (Descending)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Cost (USD)")
plt.tight_layout()
plt.show()

# =============================================================================
# Program 5: Sorting Weapons by Destruction (Descending)
# =============================================================================
sorted_by_destruction = sorted(weapons, key=lambda x: x["destruction"], reverse=True)
print("\nWeapons sorted by Destruction (Descending):")
for w in sorted_by_destruction:
    print(f"{w['name']} ({w['country']}): Destruction = {w['destruction']}")
plt.figure(figsize=(10,6))
names = [w["name"] for w in sorted_by_destruction]
destructions = [w["destruction"] for w in sorted_by_destruction]
plt.bar(names, destructions, color='lightgreen')
plt.title("Weapons Sorted by Destruction (Descending)")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Destruction")
plt.tight_layout()
plt.show()

# =============================================================================
# Program 6: Sorting by Power, Energy Output, Efficiency & Efficient Estimation of μ
# =============================================================================
sorted_by_power = sorted(weapons, key=lambda x: x["power_required"], reverse=True)
sorted_by_energy = sorted(weapons, key=lambda x: x["energy_output"], reverse=True)
sorted_by_efficiency = sorted(weapons, key=lambda x: x["efficiency"], reverse=True)

print("\nWeapons sorted by Power Required (Descending):")
for w in sorted_by_power:
    print(f"{w['name']} ({w['country']}): Power Required = {w['power_required']}")

print("\nWeapons sorted by Energy Output (Descending):")
for w in sorted_by_energy:
    print(f"{w['name']} ({w['country']}): Energy Output = {w['energy_output']}")

print("\nWeapons sorted by Efficiency (Descending):")
for w in sorted_by_efficiency:
    print(f"{w['name']} ({w['country']}): Efficiency = {w['efficiency']}")

# For efficient estimation: compute running sample mean for each weapon’s performance (μ)
def compute_running_mu(weapon, n_samples=100):
    # Simulate n_samples measurements for operational range with 5% noise
    data = np.random.normal(weapon["performance_mu"], weapon["performance_mu"] * 0.05, size=n_samples)
    running_mean = np.cumsum(data) / np.arange(1, n_samples + 1)
    return data, running_mean

fig3, ax3 = plt.subplots(figsize=(10,6))
for weapon in weapons:
    data, run_mean = compute_running_mu(weapon, n_samples=100)
    ax3.plot(run_mean, label=weapon["name"])
ax3.set_title("Running Sample Mean (Efficient Estimator of μ) for Each Weapon")
ax3.set_xlabel("Number of Samples")
ax3.set_ylabel("Estimated μ (Operational Range, km)")
ax3.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Optionally, animate the running mu for one weapon (e.g., Aegis BMD)
aegis = [w for w in weapons if w["name"] == "Aegis BMD"][0]
data_aegis, run_mean_aegis = compute_running_mu(aegis, n_samples=100)
fig4, ax4 = plt.subplots()
line4, = ax4.plot([], [], 'b-o', lw=2)
ax4.set_xlim(0, 100)
ax4.set_ylim(min(run_mean_aegis)-5, max(run_mean_aegis)+5)
ax4.set_title("Running Mean of μ for Aegis BMD")
ax4.set_xlabel("Number of Samples")
ax4.set_ylabel("Estimated μ (km)")
def update_anim(frame):
    xdata = np.arange(1, frame + 1)
    ydata = run_mean_aegis[:frame]
    line4.set_data(xdata, ydata)
    return line4,
anim_eff = FuncAnimation(fig4, update_anim, frames=range(1, 101),
                         interval=100, blit=True, repeat=False)
plt.show()