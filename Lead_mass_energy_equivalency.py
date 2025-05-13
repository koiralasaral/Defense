import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Energy released by 1 kg of Uranium (from the previous calculation)
uranium_energy = 9e16  # Joules

# Material properties for lead
lead_density = 11340  # kg/m^3
lead_strength = 50e6  # Pa (a rough estimate, lead is quite soft)

# Simplified crater formation model (very basic approximation)
# This model assumes a hemispherical crater and equates the kinetic energy
# of the ejecta to a fraction of the explosion energy.  It's a simplification!
def calculate_crater_radius(energy, material_density, material_strength):
    """
    Calculates the radius of a crater using a highly simplified model.

    Args:
        energy (float): The energy of the impact (Joules).
        material_density (float): The density of the target material (kg/m^3).
        material_strength (float): The strength of the target material (Pa).

    Returns:
        float: The radius of the crater (meters).
    """
    # This is a rudimentary approximation.  A more realistic model would
    # involve complex hydrodynamics and material science.
    crater_volume = (energy / material_strength)
    crater_radius = (3 * crater_volume / (2 * np.pi))**(1/3) # Assuming hemispherical crater
    return crater_radius

# Calculate the crater radius in lead
crater_radius_lead = calculate_crater_radius(uranium_energy, lead_density, lead_strength)

# Print the result
print(f"The estimated radius of the crater formed by 1 kg of Uranium in lead is approximately {crater_radius_lead:.2f} meters.")

# Plot the crater as a hemisphere
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Create the spherical coordinates for a hemisphere
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi/2:10j]  # Note the upper limit for v (pi/2)
x = crater_radius_lead * np.cos(u) * np.sin(v)
y = crater_radius_lead * np.sin(u) * np.sin(v)
z = crater_radius_lead * np.cos(v)

# Plot the surface of the hemisphere
ax.plot_surface(x, y, z, color='b', alpha=0.5)

# Set the axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('Estimated Crater Shape (Hemisphere)')

# Set equal aspect ratio
ax.set_xlim([-crater_radius_lead * 1.2, crater_radius_lead * 1.2])
ax.set_ylim([-crater_radius_lead * 1.2, crater_radius_lead * 1.2])
ax.set_zlim([0, crater_radius_lead * 1.2])  # Z starts at 0 for a hemisphere

plt.show()
