import folium
import matplotlib.pyplot as plt
import numpy as np
import random
import math

# --------------------------------------------
# Setup weapons (10 USA, 10 Russia)
# --------------------------------------------
weapons_usa = [f"USA_Weapon_{i}" for i in range(1, 11)]
weapons_russia = [f"Russia_Weapon_{i}" for i in range(1, 11)]

countries = {
    "USA": {
        "capital": (38.9072, -77.0369),
        "weapons": weapons_usa
    },
    "Russia": {
        "capital": (55.7558, 37.6173),
        "weapons": weapons_russia
    }
}

# --------------------------------------------
# Scatter weapon locations near capitals
# --------------------------------------------
weapon_positions = {}

for country, info in countries.items():
    base_lat, base_lon = info["capital"]
    for weapon in info["weapons"]:
        pert_lat = base_lat + random.uniform(-3, 3)
        pert_lon = base_lon + random.uniform(-3, 3)
        weapon_positions[weapon] = (pert_lat, pert_lon)

# --------------------------------------------
# Create Matplotlib plot with projectile simulation
# --------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Plot weapons
for weapon, (lat, lon) in weapon_positions.items():
    if "USA" in weapon:
        ax.plot(lon, lat, 'bo', label="USA" if "USA" not in ax.get_legend_handles_labels()[1] else "")
    else:
        ax.plot(lon, lat, 'ro', label="Russia" if "Russia" not in ax.get_legend_handles_labels()[1] else "")
    ax.text(lon + 0.2, lat + 0.2, weapon, fontsize=8)

# Simulate projectile attacks
for weapon, (lat, lon) in weapon_positions.items():
    # Random launch point
    launch_lat = lat + random.uniform(-5, 5)
    launch_lon = lon + random.uniform(-5, 5)
    
    # Generate parabolic arc
    t = np.linspace(0, 1, 100)
    x = launch_lon + (lon - launch_lon) * t
    y = launch_lat + (lat - launch_lat) * t - 10 * t * (1 - t)  # parabolic adjustment

    ax.plot(x, y, 'g--', alpha=0.7)

ax.set_title("Weapon Locations and Simulated Projectile Attacks")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.legend()
ax.grid(True)
plt.show()

# --------------------------------------------
# Create Folium map with weapon positions and attacks
# --------------------------------------------
m = folium.Map(location=[48, 0], zoom_start=2)

# Add weapons
for weapon, (lat, lon) in weapon_positions.items():
    color = "blue" if "USA" in weapon else "red"
    folium.CircleMarker(
        location=[lat, lon],
        radius=5,
        popup=weapon,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.7
    ).add_to(m)

# Simulate attack markers
for weapon, (lat, lon) in weapon_positions.items():
    launch_lat = lat + random.uniform(-5, 5)
    launch_lon = lon + random.uniform(-5, 5)

    folium.PolyLine(
        locations=[[launch_lat, launch_lon], [lat, lon]],
        color="green",
        weight=2,
        opacity=0.6
    ).add_to(m)
    
    folium.Marker(
        location=[launch_lat, launch_lon],
        icon=folium.Icon(color="green", icon="rocket", prefix='fa')
    ).add_to(m)

# Save Folium map
m.save("projectile_simulation_map.html")
print("Folium map saved as 'projectile_simulation_map.html'")
