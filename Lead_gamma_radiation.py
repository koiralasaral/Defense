import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
import math

# ================================================================
# Define materials with attenuation coefficients (μ in 1/cm)
# (These are approximate orders‐of‐magnitude for demonstration.)
# ================================================================
materials = {
    "Lead": {
        "alpha": 460.0,    # extremely strong absorption: only ~0.01 cm needed
        "beta": 0.8,       # moderate for beta particles
        "gamma": 1.2,      # gamma
        "high_gamma": 0.5  # high–energy gamma (more penetrating)
    },
    "Cement": {
        "alpha": 46.0,     # one order of magnitude lower absorption for alpha
        "beta": 0.4,
        "gamma": 0.2,
        "high_gamma": 0.1
    },
    "Reflective": {  # (e.g. polished aluminum–like)
        "alpha": 600.0,
        "beta": 0.5,
        "gamma": 0.2,
        "high_gamma": 0.15
    }
}

radiation_types = ['alpha', 'beta', 'gamma', 'high_gamma']

# ================================================================
# Compute the required thickness for 99% attenuation: 
# I/I0 = 0.01 ⇒ t_req = 4.60517 / μ.
# ================================================================
def get_required_thickness(material_name, rad_type):
    mu = materials[material_name][rad_type]
    return 4.60517 / mu

# ================================================================
# Define the geometry of the boxes.
# The inner "protected" volume is a cube of side 10 cm and height 10 cm.
# The outer box dimensions are given by adding the wall thickness (t_wall) on all sides.
# ================================================================
def get_box_bounds(inner_side=10, inner_height=10, wall_thickness=1):
    half = inner_side / 2
    # inner box: x and y from -half to half, z from 0 to inner_height.
    inner_bounds = (-half, half, -half, half, 0, inner_height)
    # outer box: extend in x & y by wall_thickness and assume the top is raised by wall_thickness.
    outer_bounds = (-half - wall_thickness, half + wall_thickness,
                    -half - wall_thickness, half + wall_thickness,
                    0, inner_height + wall_thickness)
    return inner_bounds, outer_bounds

# ================================================================
# Improved function to draw a box given bounds in 3D.
# Note: Instead of using facecolors="none", we set an explicit fully transparent color.
# ================================================================
def draw_box(ax, bounds, color='k', alpha=0.3):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    # Define the six faces using vertices.
    vertices = [
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymax, zmin), (xmin, ymax, zmin)],  # bottom
        [(xmin, ymin, zmax), (xmax, ymin, zmax), (xmax, ymax, zmax), (xmin, ymax, zmax)],  # top
        [(xmin, ymin, zmin), (xmax, ymin, zmin), (xmax, ymin, zmax), (xmin, ymin, zmax)],  # front
        [(xmin, ymax, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmin, ymax, zmax)],  # back
        [(xmin, ymin, zmin), (xmin, ymax, zmin), (xmin, ymax, zmax), (xmin, ymin, zmax)],  # left
        [(xmax, ymin, zmin), (xmax, ymax, zmin), (xmax, ymax, zmax), (xmax, ymin, zmax)]   # right
    ]
    # Create the collection, setting a transparent face color as an RGBA tuple.
    box = Poly3DCollection(vertices, edgecolors=color, alpha=alpha)
    box.set_facecolor((1, 1, 1, 0))  # fully transparent
    ax.add_collection3d(box)

# ================================================================
# Draw a static view of the shielding box with the radioactive source.
# The source (a red dot) is located at the center of the inner floor.
# The wall thickness is shown on the plot.
# ================================================================
def draw_static_box_and_source(material, rad_type, inner_side=10, inner_height=10):
    t_wall = get_required_thickness(material, rad_type)
    inner_bounds, outer_bounds = get_box_bounds(inner_side, inner_height, t_wall)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw the outer (shield) and inner (protected) boxes:
    draw_box(ax, outer_bounds, color='blue', alpha=0.3)
    draw_box(ax, inner_bounds, color='green', alpha=0.5)
    
    # Annotate the wall thickness
    wall_text = f"Wall thickness = {t_wall:.2f} cm"
    ax.text(outer_bounds[0], outer_bounds[2], outer_bounds[4] - 1, wall_text,
            color='red', fontsize=12)
    
    # Place the source at the center of the inner floor:
    source = (0, 0, 0)
    ax.scatter(*source, color='r', s=100, label="Radioactive Source")
    ax.text(0, 0, 0, " Source", color='r', fontsize=12)
    
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    ax.set_title(f"{material} Box Shielding for {rad_type} Radiation\nStatic view with Source and Box")
    
    ax.set_xlim(outer_bounds[0] - 1, outer_bounds[1] + 1)
    ax.set_ylim(outer_bounds[2] - 1, outer_bounds[3] + 1)
    ax.set_zlim(outer_bounds[4] - 1, outer_bounds[5] + 1)
    
    plt.legend()
    plt.show()
    
    return inner_bounds, outer_bounds, source, t_wall

# ================================================================
# Compute intersection of a ray with an axis‐aligned box.
# Returns the smallest positive t along the ray where the ray exits the box.
# ================================================================
def ray_box_intersection(origin, direction, bounds):
    tmin = -np.inf
    tmax = np.inf
    o = origin
    d = direction
    # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    for i, (minb, maxb) in enumerate(zip(bounds[::2], bounds[1::2])):
        if abs(d[i]) < 1e-6:
            # If d[i] is nearly zero, and the origin is not within bounds:
            if o[i] < minb or o[i] > maxb:
                return None
        else:
            t1 = (minb - o[i]) / d[i]
            t2 = (maxb - o[i]) / d[i]
            tmin = max(tmin, min(t1, t2))
            tmax = min(tmax, max(t1, t2))
    if tmax < max(0, tmin):
        return None
    return tmin if tmin > 0 else tmax

# ================================================================
# Generate a random unit vector in the upward hemisphere (z >= 0).
# ================================================================
def random_hemisphere_direction():
    phi = np.random.uniform(0, 2 * np.pi)
    cos_theta = np.random.uniform(0, 1)  # ensures z >= 0
    sin_theta = np.sqrt(1 - cos_theta ** 2)
    return np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

# ================================================================
# Animate rays from the source.
#
# Each ray is generated in a random upward direction. Its intersection
# with the inner box and then the outer box is computed. In the inner volume,
# intensity is full; in the wall region, the intensity decays as:
#   I = exp[-μ * (distance in material)].
#
# The animation shows each ray growing from the source to its exit point.
# Intermediate values for the first ray (direction, intersection distances) are printed.
# ================================================================
def animate_rays(material, rad_type, inner_bounds, outer_bounds, source, t_wall, num_rays=15):
    mu = materials[material][rad_type]
    
    # Precompute ray properties.
    rays = []
    for i in range(num_rays):
        d = random_hemisphere_direction()
        t_inner = ray_box_intersection(source, d, inner_bounds)
        t_outer = ray_box_intersection(source, d, outer_bounds)
        if t_inner is None or t_outer is None:
            t_inner, t_outer = 0.0, 0.0
        rays.append({
            'direction': d,
            't_inner': t_inner,
            't_outer': t_outer
        })
    
    # Print intermediate values for the first ray.
    if rays:
        print(f"\nExample ray for {material} with {rad_type}:")
        print(f"  Direction vector: {rays[0]['direction']}")
        print(f"  t_inner (exit inner volume): {rays[0]['t_inner']:.3f} cm")
        print(f"  t_outer (exit outer box): {rays[0]['t_outer']:.3f} cm")
    
    # Set up the 3D plot.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Draw both boxes.
    draw_box(ax, outer_bounds, color='blue', alpha=0.3)
    draw_box(ax, inner_bounds, color='green', alpha=0.5)
    
    # Plot the source.
    ax.scatter(*source, color='r', s=100, label="Source")
    
    # Initialize line objects for rays.
    ray_lines = []
    for _ in range(num_rays):
        line, = ax.plot([], [], [], lw=2)
        ray_lines.append(line)
    
    ax.set_xlabel("X (cm)")
    ax.set_ylabel("Y (cm)")
    ax.set_zlabel("Z (cm)")
    title = (f"3D Radiation Ray Animation\n"
             f"{material} Box, {rad_type} Radiation | Wall thickness = {t_wall:.2f} cm, μ = {mu}")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(outer_bounds[0] - 1, outer_bounds[1] + 1)
    ax.set_ylim(outer_bounds[2] - 1, outer_bounds[3] + 1)
    ax.set_zlim(outer_bounds[4] - 1, outer_bounds[5] + 1)
    
    num_frames = 50  # Total animation frames.

    def update(frame):
        f = frame / (num_frames - 1)
        for i, ray in enumerate(rays):
            t_total = ray['t_outer']
            current_t = f * t_total  # Distance traveled along the ray so far.
            # Full intensity inside the inner box; decay in the wall region.
            if current_t < ray['t_inner']:
                intensity = 1.0
            else:
                intensity = np.exp(-mu * (current_t - ray['t_inner']))
            endpoint = np.array(source) + ray['direction'] * current_t
            xs = [source[0], endpoint[0]]
            ys = [source[1], endpoint[1]]
            zs = [source[2], endpoint[2]]
            ray_lines[i].set_data(xs, ys)
            ray_lines[i].set_3d_properties(zs)
            # Set color with variable transparency to show intensity decay.
            ray_lines[i].set_color((1, 0, 0, intensity))
        return ray_lines

    anim = FuncAnimation(fig, update, frames=num_frames, interval=200, blit=False)
    plt.show()

# ================================================================
# Main loop: For each material and radiation type, we show:
#  1. A static 3D view of the shielding box with the radioactive source.
#  2. An animation of radiation rays emerging from the source.
# ================================================================
if __name__ == "__main__":
    for material in materials.keys():
        for rad_type in radiation_types:
            print("\n\n---------------------------------------------------")
            print(f"Simulation for Material: {material}, Radiation: {rad_type}")
            print("---------------------------------------------------")
            inner_bounds, outer_bounds, source, t_wall = draw_static_box_and_source(material, rad_type)
            animate_rays(material, rad_type, inner_bounds, outer_bounds, source, t_wall, num_rays=15)
            print("Animation complete.")
            plt.close() # Close the plot to avoid display issues in some environments.
            print()
            print("===================================================")
            