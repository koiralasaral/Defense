import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Material properties for lead
lead_density = 11340  # kg/m^3
lead_strength = 50e6  # Pa (a rough estimate, lead is quite soft)

# Simplified crater formation model (very basic approximation)
# This model assumes a hemispherical crater and equates the kinetic energy
# of the ejecta to a fraction of the explosion energy.  It's a simplification!
def calculate_crater_dimensions(energy, material_density, material_strength):
    """
    Calculates the radius and area of a crater using a highly simplified model.

    Args:
        energy (float): The energy of the impact (Joules).
        material_density (float): The density of the target material (kg/m^3).
        material_strength (float): The strength of the target material (Pa).

    Returns:
        tuple: (radius, area) of the crater in meters and square meters.
    """
    # This is a rudimentary approximation.  A more realistic model would
    # involve complex hydrodynamics and material science.
    crater_volume = (energy / material_strength)
    crater_radius = (3 * crater_volume / (2 * np.pi))**(1/3) # Assuming hemispherical crater
    crater_area = np.pi * crater_radius**2  # Area of the base of the hemisphere
    return crater_radius, crater_area

def plot_crater(radius, area, title):
    """
    Plots the crater as a hemisphere.

    Args:
        radius (float): The radius of the crater (meters).
        area (float): The area of the base of the crater (square meters).
        title (str): The title of the plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the spherical coordinates for a hemisphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]  # Note the upper limit for v (pi/2)
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)

    # Plot the surface of the hemisphere
    ax.plot_surface(x, y, z, color='b', alpha=0.5)

    # Set the axis labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Set equal aspect ratio
    ax.set_xlim([-radius * 1.2, radius * 1.2])
    ax.set_ylim([-radius * 1.2, radius * 1.2])
    ax.set_zlim([0, radius * 1.2])  # Z starts at 0 for a hemisphere

    plt.show()

def calculate_and_plot_crater(energy, material_density, material_strength, material_name, title_prefix=""):
    """
    Calculates and plots the crater dimensions, and prints the results.

    Args:
        energy (float): The energy of the impact (Joules).
        material_density (float): The density of the target material (kg/m^3).
        material_strength (float): The strength of the target material (Pa).
        material_name (str): The name of the material.
        title_prefix (str, optional): Prefix for the plot title. Defaults to "".
    """
    radius, area = calculate_crater_dimensions(energy, material_density, material_strength)
    print(f"For {material_name}:")
    print(f"  Estimated crater radius: {radius:.2f} meters")
    print(f"  Estimated crater base area: {area:.2f} square meters")
    plot_crater(radius, area, f"{title_prefix} Crater in {material_name}")

# 1. Crater from 1 kg of Uranium in Lead
uranium_energy = 9e16  # Joules
calculate_and_plot_crater(uranium_energy, lead_density, lead_strength, "Lead", "Uranium Impact")

# 2. Crater from USA and Russian weapons in Lead
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Material properties for lead
lead_density = 11340  # kg/m^3
lead_strength = 50e6  # Pa (a rough estimate, lead is quite soft)
KILOTON_TO_JOULES = 4.184e12

# Simplified crater formation model (very basic approximation)
# This model assumes a hemispherical crater and equates the kinetic energy
# of the ejecta to a fraction of the explosion energy.  It's a simplification!
def calculate_crater_dimensions(energy, material_density, material_strength):
    """
    Calculates the radius and area of a crater using a highly simplified model.

    Args:
        energy (float): The energy of the impact (Joules).
        material_density (float): The density of the target material (kg/m^3).
        material_strength (float): The strength of the target material (Pa).

    Returns:
        tuple: (radius, area) of the crater in meters and square meters.
    """
    # This is a rudimentary approximation.  A more realistic model would
    # involve complex hydrodynamics and material science.
    crater_volume = (energy / material_strength)
    crater_radius = (3 * crater_volume / (2 * np.pi))**(1/3) # Assuming hemispherical crater
    crater_area = np.pi * crater_radius**2  # Area of the base of the hemisphere
    return crater_radius, crater_area

def plot_crater(radius, area, title):
    """
    Plots the crater as a hemisphere.

    Args:
        radius (float): The radius of the crater (meters).
        area (float): The area of the base of the crater (square meters).
        title (str): The title of the plot.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Create the spherical coordinates for a hemisphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]  # Note the upper limit for v (pi/2)
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)

    # Plot the surface of the hemisphere
    ax.plot_surface(x, y, z, color='b', alpha=0.5)

    # Set the axis labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)

    # Set equal aspect ratio
    ax.set_xlim([-radius * 1.2, radius * 1.2])
    ax.set_ylim([-radius * 1.2, radius * 1.2])
    ax.set_zlim([0, radius * 1.2])  # Z starts at 0 for a hemisphere

    plt.show()

def calculate_and_plot_crater(energy, material_density, material_strength, material_name, title_prefix=""):
    """
    Calculates and plots the crater dimensions, and prints the results.

    Args:
        energy (float): The energy of the impact (Joules).
        material_density (float): The density of the target material (kg/m^3).
        material_strength (float): The strength of the target material (Pa).
        material_name (str): The name of the material.
        title_prefix (str, optional): Prefix for the plot title. Defaults to "".
    """
    radius, area = calculate_crater_dimensions(energy, material_density, material_strength)
    print(f"For {material_name}:")
    print(f"  Estimated crater radius: {radius:.2f} meters")
    print(f"  Estimated crater base area: {area:.2f} square meters")
    plot_crater(radius, area, f"{title_prefix} Crater in {material_name}")



# 2. Crater from USA and Russian weapons in Lead
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


# Calculate and plot for each weapon in the lists
for weapon in usa_weapons:
    if weapon["tnt_yield_kt"] is not None:
        energy = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
    else:
        energy = 0
    calculate_and_plot_crater(energy, lead_density, lead_strength, "Lead", f"{weapon['name']} Impact")
for weapon in russia_weapons:
    if weapon["tnt_yield_kt"] is not None:
        energy = weapon["tnt_yield_kt"] * KILOTON_TO_JOULES
    else:
        energy = 0
    calculate_and_plot_crater(energy, lead_density, lead_strength, "Lead", f"{weapon['name']} Impact")

