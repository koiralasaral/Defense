import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Weapon data with energies (in Joules)
usa_weapons = [
    {"name": "MIM-104 Patriot", "damage": 800, "tnt_yield_kt": 0.002, "range": 160, "cost": 4000000},  # Approx. 2 kg TNT equivalent (fragmentation warhead)
    {"name": "THAAD", "damage": 1200, "tnt_yield_kt": 0.009, "range": 200, "cost": 10000000}, # Approx 9kg TNT
    {"name": "Avenger", "damage": 600, "tnt_yield_kt": 0.00045, "range": 140, "cost": 2000000}, # Approx 0.45kg TNT
    {"name": "NASAMS", "damage": 900, "tnt_yield_kt": 0.011, "range": 170, "cost": 5000000},  # Approx 11kg TNT
    {"name": "Aegis BMD", "damage": 1500, "tnt_yield_kt": 0.009, "range": 220, "cost": 12000000}, # Approx 9kg TNT
    {"name": "Iron Dome", "damage": 700, "tnt_yield_kt": 0.0007, "range": 150, "cost": 3000000},  # Approx 0.7kg TNT
    {"name": "Patriot PAC-3", "damage": 1000, "tnt_yield_kt": 0.0009, "range": 180, "cost": 7000000}, # Approx 0.9kg TNT
    {"name": "Sentinel Radar", "damage": 500, "tnt_yield_kt": None, "range": 120, "cost": 1500000},  # Radar - No explosive yield
    {"name": "Arrow 3", "damage": 1300, "tnt_yield_kt": 0.009, "range": 210, "cost": 11000000}, # Approx 9kg TNT
    {"name": "SM-3", "damage": 1400, "tnt_yield_kt": 0.009, "range": 230, "cost": 13000000} # Approx 9kg TNT
]
russia_weapons = [
    {"name": "Pantsir-S1", "damage": 700, "tnt_yield_kt": 0.0002, "range": 30, "cost": 3000000}, # Approx 0.2kg
    {"name": "S-400", "damage": 1100, "tnt_yield_kt": 0.024, "range": 180, "cost": 8000000}, # Approx 24kg
    {"name": "Buk-M2", "damage": 850, "tnt_yield_kt": 0.035, "range": 100, "cost": 4000000}, # Approx 35kg
    {"name": "Tor-M2", "damage": 750, "tnt_yield_kt": 0.015, "range": 90, "cost": 2500000}, # Approx 15kg
    {"name": "S-300", "damage": 1000, "tnt_yield_kt": 0.024, "range": 190, "cost": 7000000}, # Approx 24kg
    {"name": "Iskander-M", "damage": 1500, "tnt_yield_kt": 0.48, "range": 500, "cost": 12000000},  # ~480 kg TNT
    {"name": "Kinzhal", "damage": 2000, "tnt_yield_kt": None, "range": 2000, "cost": 20000000},  # Hypersonic missile, warhead varies.
    {"name": "Kalibr", "damage": 1800, "tnt_yield_kt": 0.1, "range": 1500, "cost": 18000000}, # ~100 kg TNT
    {"name": "Bastion-P", "damage": 1300, "tnt_yield_kt": 0.2, "range": 300, "cost": 11000000}, # ~200 kg TNT
    {"name": "Sarmat", "damage": 2500, "tnt_yield_kt": 800, "range": 18000, "cost": 50000000}  # ICBM, ~800,000 kg TNT (0.8 megatons)
]

# Material data with resistance factors (higher value means more resistance)
materials = {
    "Clay": 1.4,
    "Rock": 2.7,
    "Lead": 8,
    "Alluvial Soil": 1.8
}

# Conversion factor: 1 kiloton of TNT in Joules
KILOTON_TO_JOULES = 4.184e12

def calculate_blast_radius(energy, material_resistance):
    """
    Calculates the blast radius based on energy and material resistance.
    A simplified model is used: radius = (energy / resistance)^(1/3)
    """
    return (energy / material_resistance) ** (1/3)

def plot_blast_sphere(radius, material_name, weapon_name):
    """
    Plots a 3D sphere representing the blast radius.

    Args:
        radius (float): The radius of the sphere.
        material_name (str): The name of the material.
        weapon_name (str): The name of the weapon.
    """
    # Create a figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create the spherical coordinates
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)

    # Plot the sphere
    ax.plot_surface(x, y, z, color='r', alpha=0.5)

    # Set the plot title and labels
    ax.set_title(f"{weapon_name} in {material_name}\nBlast Radius: {radius:.2f} m")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Set equal aspect ratio
    ax.set_xlim([-radius * 1.2, radius * 1.2])
    ax.set_ylim([-radius * 1.2, radius * 1.2])
    ax.set_zlim([-radius * 1.2, radius * 1.2])
    
    # Remove the background color for better visualization
    ax.set_facecolor('none')

    plt.show()

def plot_weapons_data(weapons_data, title, material_name_filter=None):
    """
    Plots blast spheres for each weapon in the given dataset, optionally filtered by material.

    Args:
        weapons_data (list): A list of weapon dictionaries.
        title (str): The title of the plot.
        material_name_filter (str, optional): The name of the material to filter by.
            If None, all materials are plotted. Defaults to None.
    """
    # Iterate over weapons and materials to generate plots
    for weapon in weapons_data:
        for material_name, material_resistance in materials.items():
            if material_name_filter is None or material_name == material_name_filter:
                # Calculate the blast radius
                if weapon["tnt_yield_kt"] is not None:
                    energy = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
                else:
                    energy = 0 # if there is no TNT yield, the energy is zero.
                blast_radius = calculate_blast_radius(energy, material_resistance)
                
                # Plot the blast sphere
                plot_blast_sphere(blast_radius, material_name, weapon["name"])

def calculate_and_print_blast_data(weapons_data, country_name):
    """
    Calculates and prints blast radius data for a list of weapons.

    Args:
        weapons_data (list): A list of weapon dictionaries.
        country_name (str): The name of the country (e.g., "USA" or "Russia").
    """
    blast_radii = []
    for weapon in weapons_data:
        # Calculate blast radius for a standard material (e.g., Clay)
        if weapon["tnt_yield_kt"] is not None:
            energy = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
        else:
            energy = 0
        blast_radius = calculate_blast_radius(energy, materials["Clay"])
        blast_radii.append(blast_radius)
    
    print(f"\n{country_name} Weapons Blast Radii (in meters):")
    print(blast_radii)
    
    # Calculate and print partial sums
    partial_sums = []
    current_sum = 0
    for radius in blast_radii:
        current_sum += radius
        partial_sums.append(current_sum)
        print(f"Partial Sum: {current_sum:.2f}")
    
    # Estimate the limit of the sequence (if it exists)
    if len(partial_sums) > 1:
        last_few_sums = partial_sums[-5:] if len(partial_sums) > 5 else partial_sums
        if all(abs(last_few_sums[i] - last_few_sums[i-1]) < 0.1 for i in range(1, len(last_few_sums))):
            print(f"Estimated Limit of Partial Sums: {partial_sums[-1]:.2f}")
        else:
            print("The partial sums do not appear to converge to a limit.")
    else:
        print("Not enough data to estimate the limit.")
    

# Update weapon data with energy from TNT yield
for weapon in usa_weapons:
    if weapon["tnt_yield_kt"] is not None:
        weapon["energy"] = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
    else:
        weapon["energy"] = 0  # Set energy to 0 for weapons without TNT yield
for weapon in russia_weapons:
    if weapon["tnt_yield_kt"] is not None:
        weapon["energy"] = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
    else:
        weapon["energy"] = 0

# Plot data for USA and Russia for Lead
plot_weapons_data(usa_weapons, "Blast Radius for USA Weapons in Lead", material_name_filter="Lead")
plot_weapons_data(russia_weapons, "Blast Radius for Russian Weapons in Lead", material_name_filter="Lead")

# Calculate and print blast radius data
calculate_and_print_blast_data(usa_weapons, "USA")
calculate_and_print_blast_data(russia_weapons, "Russia")
