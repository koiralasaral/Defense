import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
from matplotlib.animation import FuncAnimation

# Ellipse Parameters
a = 5  # Semi-major axis
b = 3  # Semi-minor axis
phi_offset = np.radians(40)  # Angular offset between φ and φ'

# Generate fixed data for ellipse and auxiliary circle (z = 0 for both)
theta = np.linspace(0, 2 * np.pi, 400)
ellipse_x = a * np.cos(theta)
ellipse_y = b * np.sin(theta)
ellipse_z = np.zeros_like(theta)

aux_circle_x = a * np.cos(theta)
aux_circle_y = a * np.sin(theta)
aux_circle_z = np.zeros_like(theta)

# Set up the 3D figure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(-a - 1, a + 1)
ax.set_ylim(-a - 1, a + 1)
ax.set_zlim(-b - 1, b + 1)
ax.set_title('3D Animation: Ellipse with Auxiliary Circle')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Plot fixed elements: ellipse and auxiliary circle
ax.plot(ellipse_x, ellipse_y, ellipse_z, 'b-', label='Ellipse')
ax.plot(aux_circle_x, aux_circle_y, aux_circle_z, 'g--', label='Auxiliary Circle')

# Create animated elements; note that we use empty lists for initialization.
point_ellipse1, = ax.plot([], [], [], 'ro', label='Point 1 (Ellipse)')
point_ellipse2, = ax.plot([], [], [], 'ro', label='Point 2 (Ellipse)')
chord_line, = ax.plot([], [], [], 'm-', linewidth=2, label='Chord')
point_aux1, = ax.plot([], [], [], 'ko', label='Point 1 (Auxiliary Circle)')
point_aux2, = ax.plot([], [], [], 'ko', label='Point 2 (Auxiliary Circle)')

def init():
    point_ellipse1.set_data([], [])
    point_ellipse1.set_3d_properties([])
    point_ellipse2.set_data([], [])
    point_ellipse2.set_3d_properties([])
    chord_line.set_data([], [])
    chord_line.set_3d_properties([])
    point_aux1.set_data([], [])
    point_aux1.set_3d_properties([])
    point_aux2.set_data([], [])
    point_aux2.set_3d_properties([])
    return point_ellipse1, point_ellipse2, chord_line, point_aux1, point_aux2

def update(frame):
    # Animate the eccentric angles
    phi1 = frame * 0.05
    phi2 = (phi1 + phi_offset) % (2 * np.pi)

    # Calculate the ellipse coordinates (with z = 0 because our curve is in the xy-plane)
    x1, y1, z1 = a * np.cos(phi1), b * np.sin(phi1), 0
    x2, y2, z2 = a * np.cos(phi2), b * np.sin(phi2), 0

    # Calculate the corresponding auxiliary circle coordinates (radius = a)
    aux_x1, aux_y1, aux_z1 = a * np.cos(phi1), a * np.sin(phi1), 0
    aux_x2, aux_y2, aux_z2 = a * np.cos(phi2), a * np.sin(phi2), 0

    # Set the updated data as sequences
    point_ellipse1.set_data([x1], [y1])
    point_ellipse1.set_3d_properties([z1])
    point_ellipse2.set_data([x2], [y2])
    point_ellipse2.set_3d_properties([z2])
    
    chord_line.set_data([x1, x2], [y1, y2])
    chord_line.set_3d_properties([z1, z2])
    
    point_aux1.set_data([aux_x1], [aux_y1])
    point_aux1.set_3d_properties([aux_z1])
    point_aux2.set_data([aux_x2], [aux_y2])
    point_aux2.set_3d_properties([aux_z2])

    # Also update the camera (azimuth rotates with the frame)
    ax.view_init(elev=20, azim=frame)

    return point_ellipse1, point_ellipse2, chord_line, point_aux1, point_aux2

# Create the 3D animation
anim3d = FuncAnimation(fig, update, frames=360, init_func=init,
                       interval=50, blit=False)
plt.legend()
plt.show()