import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Weapon data with energies (in Joules)
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

# Material data with resistance factors (higher value means more resistance)
materials = {
    "Clay": 1.4,
    "Rock": 2.7,
    "Lead": 8,
    "Alluvial Soil": 1.8
}

def calculate_blast_radius(energy, material_resistance):
    """
    Calculates the blast radius based on energy and material resistance.
    A simplified model is used: radius = (energy / material_resistance)^(1/3)
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

def plot_weapons_data(weapons_data, title):
    """
    Plots blast spheres for each weapon in the given dataset.

    Args:
        weapons_data (list): A list of weapon dictionaries.
        title (str): The title of the plot.
    """
    # Iterate over weapons and materials to generate plots
    for weapon in weapons_data:
        for material_name, material_resistance in materials.items():
            # Calculate the blast radius
            blast_radius = calculate_blast_radius(weapon["energy"], material_resistance)
            
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
        blast_radius = calculate_blast_radius(weapon["energy"], materials["Clay"])
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

# Plot data for USA and Russia
plot_weapons_data(usa_weapons, "Blast Radius for USA Weapons")
plot_weapons_data(russia_weapons, "Blast Radius for Russian Weapons")

# Calculate and print blast radius data
calculate_and_print_blast_data(usa_weapons, "USA")
calculate_and_print_blast_data(russia_weapons, "Russia")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Weapon data with energies (in Joules)
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

# Material data with resistance factors (higher value means more resistance)
materials = {
    "Clay": 1.4,
    "Rock": 2.7,
    "Lead": 8,
    "Alluvial Soil": 1.8
}

def calculate_blast_radius(energy, material_resistance):
    """
    Calculates the blast radius based on energy and material resistance.
    A simplified model is used: radius = (energy / material_resistance)^(1/3)
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
                blast_radius = calculate_blast_radius(weapon["energy"], material_resistance)
                
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
        blast_radius = calculate_blast_radius(weapon["energy"], materials["Clay"])
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

# Plot data for USA and Russia for Lead
plot_weapons_data(usa_weapons, "Blast Radius for USA Weapons in Lead", material_name_filter="Lead")
plot_weapons_data(russia_weapons, "Blast Radius for Russian Weapons in Lead", material_name_filter="Lead")

# Calculate and print blast radius data
calculate_and_print_blast_data(usa_weapons, "USA")
calculate_and_print_blast_data(russia_weapons, "Russia")
