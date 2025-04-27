import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import folium
import math

# --- Weapon data (from provided script) ---
weapons = [
    {"name": "MIM-104 Patriot", "country": "USA",   "weight": 5000,  "cost":  4000000,
     "destruction": 800,  "power_required": 100, "energy_output": 300, "efficiency": 0.85},
    {"name": "THAAD",           "country": "USA",   "weight": 20000, "cost": 10000000,
     "destruction": 1200, "power_required": 150, "energy_output": 350, "efficiency": 0.88},
    {"name": "Avenger",         "country": "USA",   "weight": 3000,  "cost":  2000000,
     "destruction": 600,  "power_required": 80,  "energy_output": 250, "efficiency": 0.80},
    {"name": "NASAMS",          "country": "USA",   "weight": 7000,  "cost":  5000000,
     "destruction": 900,  "power_required": 110, "energy_output": 280, "efficiency": 0.86},
    {"name": "Aegis BMD",       "country": "USA",   "weight": 25000, "cost": 12000000,
     "destruction": 1500, "power_required": 200, "energy_output": 400, "efficiency": 0.90},
    {"name": "Pantsir-S1",      "country": "Russia","weight": 6000,  "cost":  3000000,
     "destruction": 700,  "power_required": 90,  "energy_output": 260, "efficiency": 0.82},
    {"name": "S-400",           "country": "Russia","weight": 15000, "cost": 8000000,
     "destruction": 1100, "power_required": 130, "energy_output": 330, "efficiency": 0.87},
    {"name": "Buk-M2",          "country": "Russia","weight": 8000,  "cost":  4000000,
     "destruction": 850,  "power_required": 95,  "energy_output": 270, "efficiency": 0.83},
    {"name": "Tor-M2",          "country": "Russia","weight": 4000,  "cost":  2500000,
     "destruction": 750,  "power_required": 85,  "energy_output": 240, "efficiency": 0.81},
    {"name": "S-300",           "country": "Russia","weight": 13000, "cost": 7000000,
     "destruction": 1000, "power_required": 120, "energy_output": 320, "efficiency": 0.86}
]

# Prepare linear coefficients and bounds
power_req = np.array([w["power_required"] for w in weapons])
destr = np.array([w["destruction"] for w in weapons])
costs = np.array([w["cost"] for w in weapons])
bounds = [(0, 50) for _ in weapons]  # 0 <= x_i <= 50

# Print the linear objective function formulas
names = [w["name"] for w in weapons]
varnames = [name.replace('-', '').replace(' ', '_') for name in names]
E_terms = " + ".join(f"{w['power_required']}*x_{var}" for w, var in zip(weapons, varnames))
D_terms = " + ".join(f"{w['destruction']}*x_{var}" for w, var in zip(weapons, varnames))
C_terms = " + ".join(f"{w['cost']}*x_{var}" for w, var in zip(weapons, varnames))
print("Objective functions:")
print(f"  Minimize Energy:   E_total = {E_terms}")
print(f"  Maximize Destruct: D_total = {D_terms}")
print(f"  Minimize Cost:     C_total = {C_terms}\n")

# Constraint: total destruction >= threshold
threshold = 10000  # as in the Monte Carlo simulation&#8203;:contentReference[oaicite:5]{index=5}
A_ub = -destr.reshape(1, -1)
b_ub = np.array([-threshold])

# (i) Minimize total energy required
res_energy = linprog(c=power_req, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print("Minimize Energy Required (with destruction >= 10000):")
print("  Optimal Energy = {:.2f}".format(res_energy.fun))
for xi, nm in zip(res_energy.x, names):
    if xi > 1e-6:
        print(f"   {nm}: {xi:.2f}")

# (ii) Maximize total destruction (no extra constraint other than bounds)
res_destr = linprog(c=-destr, bounds=bounds, method='highs')
print("\nMaximize Total Destruction:")
print("  Optimal Destruction = {:.0f}".format(-res_destr.fun))
for xi, nm in zip(res_destr.x, names):
    if xi > 1e-6:
        print(f"   {nm}: {xi:.0f}")

# (iii) Minimize total cost
res_cost = linprog(c=costs, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print("\nMinimize Total Cost (with destruction >= 10000):")
print("  Optimal Cost = {:.2f}".format(res_cost.fun))
for xi, nm in zip(res_cost.x, names):
    if xi > 1e-6:
        print(f"   {nm}: {xi:.2f}")

# --- Simulate movement of target coordinates ---
# Start (Brest) and target (Washington, DC) coordinates
start = (53.3108, 28.8103)  # 53°18'38.9"N 28°48'37.1"E (approx.)
target = (38.9072, -77.0369)  # Washington, DC&#8203;:contentReference[oaicite:6]{index=6}
def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius (km)
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

# Define a sequence of strikes (from high to lower destruction) until reaching DC
sequence = ["Aegis BMD", "S-400", "MIM-104 Patriot", "THAAD", 
            "S-300", "NASAMS", "Buk-M2", "Pantsir-S1", "Tor-M2"]
coords = [start]
current = start
for weapon_name in sequence:
    dmg = next(w["destruction"] for w in weapons if w["name"] == weapon_name)
    dist = haversine(current[0], current[1], target[0], target[1])
    if dmg >= dist:
        coords.append(target)
        break
    frac = dmg / dist
    new_lat = current[0] + frac * (target[0] - current[0])
    new_lon = current[1] + frac * (target[1] - current[1])
    current = (new_lat, new_lon)
    coords.append(current)

print("\nSimulated strikes and new coordinates:")
for i, (lat, lon) in enumerate(coords):
    print(f"  Step {i}: lat={lat:.4f}, lon={lon:.4f}")

# --- Plot movement path using Matplotlib ---
lats = [c[0] for c in coords]
lons = [c[1] for c in coords]
plt.figure(figsize=(6,6))
plt.plot(lons, lats, 'ro-')
for i, (lat, lon) in enumerate(coords):
    plt.text(lon, lat, str(i), fontsize=8, ha='right')
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Movement Path of Target (Brest to Washington, DC)")
plt.grid(True)
plt.savefig("movement_path.png")
plt.show()

# --- Folium map of movement ---
m3 = folium.Map(location=[(start[0]+target[0])/2, (start[1]+target[1])/2], zoom_start=3)
# Mark start and target
folium.Marker(location=start, popup="Start (Brest)", icon=folium.Icon(color='green')).add_to(m3)
folium.Marker(location=target, popup="Washington, DC", icon=folium.Icon(color='blue')).add_to(m3)
# Mark intermediate steps
for i, (lat, lon) in enumerate(coords[1:-1], start=1):
    folium.Marker(location=(lat, lon), popup=f"Step {i}", icon=folium.Icon(color='red')).add_to(m3)
# Draw polyline path
folium.PolyLine(locations=coords, color="red", weight=2.5, opacity=0.8).add_to(m3)
m3.save("movement_path_map.html")
print("\nMovement path map saved as 'movement_path_map.html'.")
