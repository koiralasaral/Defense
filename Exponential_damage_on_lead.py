import numpy as np
import matplotlib.pyplot as plt

# --- Rocket Data (as provided) ---
rocket_data = {
    "V2": {
        "range_km": 320,
        "tnt_equivalent_kg": 1000,
        "cd": 0.15,
        "speed_km_s": 1.6,
        "length_m": 14,
        "diameter_m": 1.65,
        "type": "Ballistic",
        "cost_million_usd": 0.3,
        "energy_joules": 1000 * 4.184e6,
        "mass_kg": 12900,  # Actual V2 mass
        "warhead_mass_kg": 1000
    },
    "Iskander": {
        "range_km": 500,
        "tnt_equivalent_kg": 480,
        "cd": 0.30,
        "speed_km_s": 2.1,
        "length_m": 7.3,
        "diameter_m": 0.92,
        "type": "Ballistic",
        "cost_million_usd": 3,
        "energy_joules": 480 * 4.184e6,
        "mass_kg": 3800,
        "warhead_mass_kg": 480
    },
    "Tomahawk": {
        "range_km": 2500,
        "tnt_equivalent_kg": 454,
        "cd": 0.30,
        "speed_km_s": 0.24,
        "length_m": 6.25,
        "diameter_m": 0.52,
        "type": "Cruise",
        "cost_million_usd": 1.5,
        "energy_joules": 454 * 4.184e6,
        "mass_kg": 1200,
        "warhead_mass_kg": 454
    },
    "Minuteman_III": {
        "range_km": 13000,
        "tnt_equivalent_kg": 300000,
        "cd": 0.20,
        "speed_km_s": 7,
        "length_m": 18.2,
        "diameter_m": 1.67,
        "type": "ICBM",
        "cost_million_usd": 7,
        "energy_joules": 300000 * 4.184e6,
        "mass_kg": 35400,
        "warhead_mass_kg": 300000
    },
    "Topol_M": {
        "range_km": 11000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.22,
        "speed_km_s": 7.3,
        "length_m": 22.7,
        "diameter_m": 1.86,
        "type": "ICBM",
        "cost_million_usd": 8,
        "energy_joules": 1000000 * 4.184e6,
        "mass_kg": 47200,
        "warhead_mass_kg": 1000000
    },
    "DF_41": {
        "range_km": 14000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.21,
        "speed_km_s": 7.8,
        "length_m": 16.5,
        "diameter_m": 2,
        "type": "ICBM",
        "cost_million_usd": 10,
        "energy_joules": 1000000 * 4.184e6,
        "mass_kg": 52000,
        "warhead_mass_kg": 1000000
    },
    "Sarmat": {
        "range_km": 18000,
        "tnt_equivalent_kg": 8000000,
        "cd": 0.23,
        "speed_km_s": 7.5,
        "length_m": 35.5,
        "diameter_m": 3,
        "type": "ICBM",
        "cost_million_usd": 20,
        "energy_joules": 8000000 * 4.184e6,
        "mass_kg": 208000,
        "warhead_mass_kg": 8000000
    }
}

# --- Function to compute damage on lead ---
def compute_damage_in_lead(rocket):
    """
    Compute the crater area (damage on lead in m²) using:
       - KE = 0.5 * warhead_mass * v^2, with v in m/s
       - total energy = KE + explosive energy (given in energy_joules)
       - energy (in kt TNT) = total_energy / (4.184e12)
       - crater_diameter (km) = 0.1 * (energy_kt ** (1/3.4)) * (200e6/12e6)**(1/3)
       - crater area = pi * ( (crater_diameter*1000/2) )²
    (We use 12e6 as the compressive strength for lead.)
    """
    # Impact speed in m/s
    v = rocket['speed_km_s'] * 1000  
    ke = 0.5 * rocket['warhead_mass_kg'] * v**2
    explosive_energy = rocket['energy_joules']
    total_energy = ke + explosive_energy
    energy_kt = total_energy / (4.184e12)
    
    # Scaling factor for compressive strength of lead (12e6 Pa)
    cs_factor = (200e6 / 12e6)**(1/3)
    crater_diameter_km = 0.1 * (energy_kt ** (1/3.4)) * cs_factor
    crater_diameter_m = crater_diameter_km * 1000
    area = np.pi * (crater_diameter_m / 2)**2  # in m²
    return area

# --- Compute damage for each rocket ---
rockets = ['V2', 'Iskander', 'Tomahawk', 'Topol_M', 'DF_41', 'Sarmat']
damage = {}
N_values = {}

for r in rockets:
    damage[r] = compute_damage_in_lead(rocket_data[r])
    
# Use V2 as the baseline damage
damage_V2 = damage['V2']

# --- Define the exponential function ---
# We choose mu = 0 and sigma = 1 so that:
#    f(x) = coefficient * exp((x - mu)/sigma) = damage_V2 * exp(x)
coeff = damage_V2
mu = 0
sigma = 1

def f(x):
    return coeff * np.exp(x)

# --- Calculate N for each rocket ---
# We define N such that f(N) equals the damage on lead.
# That is, N = ln(damage(rocket) / damage(V2)), so that f(N)=damage(V2)*exp(N)=damage(rocket)
for r in rockets:
    if r == 'V2':
        N_values[r] = 0.0
    else:
        N_values[r] = np.log(damage[r] / damage_V2)

# --- Print the results ---
print("Rocket      Damage on Lead (m^2)      N value             f(N)=damage(V2)*exp(N)")
for r in rockets:
    f_val = f(N_values[r])
    print(f"{r:10s} | {damage[r]:18.2f} | {N_values[r]:+10.4f} | {f_val:18.2f}")

# --- Plot the exponential function and rocket data points ---
# Create a range of x values that covers the computed N values
x_vals = np.linspace(min(N_values.values()) - 0.5, max(N_values.values()) + 0.5, 200)
y_vals = f(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'b-', lw=2,
         label="Exponential Model: f(x)=damage(V2)*exp(x)")

# Plot each rocket's point (marker and text)
for r in rockets:
    plt.scatter(N_values[r], damage[r], label=f"{r}", s=80)
    plt.text(N_values[r], damage[r], f"  {r}", fontsize=9, va='bottom')

plt.xlabel("N (ln(damage / damage(V2)))")
plt.ylabel("Damage on Lead (m²)")
plt.title("Exponential Scaling of Damage on Lead Compared to V2")
plt.legend()
plt.grid(True, which='both', ls='--', alpha=0.7)
plt.show()
