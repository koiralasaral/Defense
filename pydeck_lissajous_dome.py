import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import folium
import pydeck as pdk
import sympy as sp

# =============================================================================
# I. Define the Two Given Points (in decimal degrees)
# =============================================================================
# Point 1: 53°15'27.0"N, 8°41'22.9"E → (lat, lon) ≈ (53.2575, 8.6897)
# Point 2: 51°35'24.4"N, 6°09'03.6"E → (lat, lon) ≈ (51.5901, 6.1510)
p1 = {"lat": 53.2575, "lon": 8.6897}
p2 = {"lat": 51.5901, "lon": 6.1510}

# =============================================================================
# II. Define the Lissajous Curve and Its Derivatives
# =============================================================================
# We use a parametric form:
#   x(t) = A*sin(t + δ) + C      (x = longitude)
#   y(t) = B*sin(t)      + D      (y = latitude)
#
# For t = 0 we require:
#   x(0) = A*sin(δ) + C = p1["lon"] = 8.6897
#   y(0) = D = p1["lat"] = 53.2575
#
# For t = π/2 we require:
#   x(π/2) = A*cos(δ) + C = p2["lon"] = 6.1510
#   y(π/2) = B + D = p2["lat"] = 51.5901
#
# Set:
D = p1["lat"]                    # 53.2575
B = p2["lat"] - D                # 51.5901 - 53.2575 = -1.6674
δ = -np.pi/4                     # chosen for convenience (sin(–π/4)≈–0.7071, cos(–π/4)≈ 0.7071)
# Then we have:
#   A*(-0.7071) + C = 8.6897  and A*(0.7071) + C = 6.1510.
# Subtracting: –1.4142 * A = 2.5387 → A ≈ –1.795;
# Then C = 6.1510 – A*(0.7071) ≈ 7.421.
A = -1.795
C = 6.1510 - (A * 0.7071)        # ≈ 7.421

def lissajous(t):
    """Return (longitude, latitude) on the curve for parameter t."""
    x = A * np.sin(t + δ) + C
    y = B * np.sin(t) + D
    return x, y

def lissajous_derivatives(t):
    """Return first derivatives (dx/dt, dy/dt) at t."""
    dx = A * np.cos(t + δ)
    dy = B * np.cos(t)
    return dx, dy

def lissajous_second_deriv(t):
    """Return second derivatives (d²x/dt², d²y/dt²) at t."""
    ddx = -A * np.sin(t + δ)
    ddy = -B * np.sin(t)
    return ddx, ddy

def osculating_circle(t, num_points=100):
    """
    Compute the osculating circle at parameter t.
    Returns a tuple (center_x, center_y, R) and arrays of points (circle_x, circle_y)
    around the circle.
    """
    x, y = lissajous(t)
    dx, dy = lissajous_derivatives(t)
    ddx, ddy = lissajous_second_deriv(t)
    num = abs(dx * ddy - dy * ddx)
    den = (dx**2 + dy**2)**1.5
    κ = num/den if den != 0 else 0
    R = 1/κ if κ != 0 else 1e6  # large R if curvature is near zero
    speed = np.hypot(dx, dy)
    T = np.array([dx, dy]) / speed
    N = np.array([-T[1], T[0]])  # unit normal (rotation by –90°)
    center = np.array([x, y]) + R * N
    theta = np.linspace(0, 2*np.pi, num_points)
    circle_x = center[0] + R * np.cos(theta)
    circle_y = center[1] + R * np.sin(theta)
    return (center[0], center[1], R), circle_x, circle_y

# For illustration, let’s sample additional points along the curve:
# We already have: t = 0 (p1) and t = π/2 (p2)
# Also note:
#   t = π gives (≈ (6.1510, 53.2575))
#   t = 3π/2 gives (≈ (8.6909, 54.9249))
#   t = 2π returns back near p1.
sample_t = [0, np.pi/2, np.pi, 3*np.pi/2]
other_points = [{"t": t, "coord": lissajous(t)} for t in sample_t]
print("Sampled Lissajous Curve Points:")
for pt in other_points:
    print(f"t = {pt['t']:.2f}  →  (lon, lat) = ({pt['coord'][0]:.4f}, {pt['coord'][1]:.4f})")

# =============================================================================
# III. Matplotlib Animation: Lissajous Curve & Osculating Circle
# =============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Lissajous Curve with Osculating Circle")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
ax.set_xlim(5, 10)
ax.set_ylim(50, 55)
# Plot the fixed given points:
ax.plot(p1["lon"], p1["lat"], 'ro', label="Point 1")
ax.plot(p2["lon"], p2["lat"], 'bo', label="Point 2")
(line_curve,) = ax.plot([], [], 'k-', lw=2, label="Lissajous curve")
(point_marker,) = ax.plot([], [], 'ko', markersize=8, label="Current point")
(line_circle,) = ax.plot([], [], 'm--', lw=1, label="Osculating circle")
ax.legend(loc="upper right")

t_vals = np.linspace(0, 2*np.pi, 400)
def init():
    line_curve.set_data([], [])
    point_marker.set_data([], [])
    line_circle.set_data([], [])
    return line_curve, point_marker, line_circle

def animate(i):
    t_current = t_vals[i]
    xs, ys = lissajous(t_vals[:i+1])
    line_curve.set_data(xs, ys)
    x_curr, y_curr = lissajous(t_current)
    point_marker.set_data([x_curr], [y_curr])
    (cx, cy, R), circ_x, circ_y = osculating_circle(t_current)
    line_circle.set_data(circ_x, circ_y)
    return line_curve, point_marker, line_circle

ani = animation.FuncAnimation(fig, animate, frames=len(t_vals),
                              init_func=init, interval=20, blit=True)
plt.show()

# =============================================================================
# IV. Folium Map: Lissajous Curve & 50 km "Domes" (2D Overlays)
# =============================================================================
# Sample many points for the polyline.
t_sample = np.linspace(0, 2*np.pi, 300)
liss_x, liss_y = lissajous(t_sample)
# Folium order: [lat, lon]
liss_points = [[y, x] for x, y in zip(liss_x, liss_y)]

# Also prepare markers for the key sample points.
marker_points = []
for pt in other_points:
    lon, lat = pt["coord"]
    marker_points.append({"lat": lat, "lon": lon, "popup": f"t = {pt['t']:.2f}"})

center_lat = (p1["lat"] + p2["lat"]) / 2
center_lon = (p1["lon"] + p2["lon"]) / 2
fol_map = folium.Map(location=[center_lat, center_lon], zoom_start=7)
folium.TileLayer("OpenStreetMap", name="Classic").add_to(fol_map)
folium.TileLayer("Esri.WorldImagery", name="Satellite").add_to(fol_map)

# Add markers for the key sample points.
for mp in marker_points:
    folium.Marker(
        location=[mp["lat"], mp["lon"]],
        popup=mp["popup"],
        tooltip=f"t = {mp['popup']}"
    ).add_to(fol_map)

# Add the full Lissajous curve as a polyline.
folium.PolyLine(liss_points, color="red", weight=3, tooltip="Lissajous Curve").add_to(fol_map)

# Add a 50 km circle overlay (a 2D dome) at each key point.
for mp in marker_points:
    folium.Circle(
        location=[mp["lat"], mp["lon"]],
        radius=50000,   # 50 km
        color="blue",
        fill=True,
        fill_opacity=0.2,
        tooltip="50 km Dome"
    ).add_to(fol_map)

folium.LayerControl().add_to(fol_map)
fol_map.save("folium_lissajous_domes.html")
print("Folium map saved as 'folium_lissajous_domes.html'.")

# =============================================================================
# V. Pydeck 3D Map: Geodesic Domes (50 km radius) Above Each Point
# =============================================================================
# We “drape” over each key point a 3D geodesic dome. The dome is built by
# subdividing an icosahedron and then scaling it to have a 50 km horizontal radius.
def create_icosahedron():
    phi = (1 + np.sqrt(5)) / 2.0
    vertices = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi],
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1]
    ], dtype=float)
    vertices /= np.linalg.norm(vertices[0])
    faces = [
        (0,11,5), (0,5,1), (0,1,7), (0,7,10), (0,10,11),
        (1,5,9), (5,11,4), (11,10,2), (10,7,6), (7,1,8),
        (3,9,4), (3,4,2), (3,2,6), (3,6,8), (3,8,9),
        (4,9,5), (2,4,11), (6,2,10), (8,6,7), (9,8,1)
    ]
    return vertices, faces

def subdivide_face(v1, v2, v3, frequency):
    grid = {}
    for i in range(frequency+1):
        for j in range(frequency+1-i):
            u = i / frequency
            v = j / frequency
            w = 1 - u - v
            point = w*v1 + u*v2 + v*v3
            point /= np.linalg.norm(point)
            grid[(i, j)] = point
    triangles = []
    for i in range(frequency):
        for j in range(frequency-i):
            pt1 = grid[(i, j)]
            pt2 = grid[(i+1, j)]
            pt3 = grid[(i, j+1)]
            triangles.append((pt1, pt2, pt3))
            if j < frequency-i-1:
                pt4 = grid[(i+1, j+1)]
                triangles.append((pt2, pt4, pt3))
    return triangles

def create_geodesic_dome(frequency, dome_only=True):
    vertices, faces = create_icosahedron()
    all_triangles = []
    for face in faces:
        all_triangles.extend(subdivide_face(vertices[face[0]], vertices[face[1]], vertices[face[2]], frequency))
    if dome_only:
        all_triangles = [tri for tri in all_triangles if all(pt[2] >= 0 for pt in tri)]
    return all_triangles

dome_freq = 3
dome_radius = 50000  # 50 km in meters
dome_triangles = create_geodesic_dome(dome_freq, dome_only=True)

def convert_vertex_to_coords(airport_lat, airport_lon, vertex, dome_radius):
    # Convert a vertex on a unit sphere to (lon, lat, altitude) assuming
    # 1° latitude ≈ 111000 m and longitude adjusted by cos(lat).
    dlon = (vertex[0] * dome_radius) / (111000 * math.cos(math.radians(airport_lat)))
    dlat = (vertex[1] * dome_radius) / 111000
    alt  = vertex[2] * dome_radius
    return [airport_lon + dlon, airport_lat + dlat, alt]

pydeck_dome_lines = []
# Use the key sample points (from marker_points) for the dome overlays.
for mp in marker_points:
    for tri in dome_triangles:
        v0 = convert_vertex_to_coords(mp["lat"], mp["lon"], tri[0], dome_radius)
        v1 = convert_vertex_to_coords(mp["lat"], mp["lon"], tri[1], dome_radius)
        v2 = convert_vertex_to_coords(mp["lat"], mp["lon"], tri[2], dome_radius)
        pydeck_dome_lines.append({"path": [v0, v1], "point": f"{mp['lat']:.4f},{mp['lon']:.4f}"})
        pydeck_dome_lines.append({"path": [v1, v2], "point": f"{mp['lat']:.4f},{mp['lon']:.4f}"})
        pydeck_dome_lines.append({"path": [v2, v0], "point": f"{mp['lat']:.4f},{mp['lon']:.4f}"})

airport_layer = pdk.Layer(
    "ScatterplotLayer",
    data=[{"lat": mp["lat"], "lon": mp["lon"], "name": f"t={mp['popup']}"} for mp in marker_points],
    get_position=["lon", "lat"],
    get_radius=2000,
    get_fill_color=[255, 0, 0],
    pickable=True
)

dome_layer_pydeck = pdk.Layer(
    "LineLayer",
    data=pydeck_dome_lines,
    get_source_position="path[0]",
    get_target_position="path[1]",
    get_color="[0, 0, 0]",
    width_min_pixels=2,
    pickable=True
)

# (Optional) Show the Lissajous curve as a 3D path (at altitude 0).
lissajous_points = []
for t in np.linspace(0, 2*np.pi, 200):
    x, y = lissajous(t)
    lissajous_points.append([x, y, 0])
lissajous_layer = pdk.Layer(
    "PathLayer",
    data=[{"path": lissajous_points, "name": "Lissajous Curve"}],
    get_path="path",
    get_color="[255, 0, 0]",
    width_scale=10,
    width_min_pixels=2,
    pickable=True
)

view_state = pdk.ViewState(
    latitude=center_lat,
    longitude=center_lon,
    zoom=7,
    pitch=50,
    bearing=0
)

deck = pdk.Deck(
    layers=[airport_layer, dome_layer_pydeck, lissajous_layer],
    initial_view_state=view_state,
    tooltip={"text": "{name}"}
)

deck.to_html("pydeck_lissajous_domes.html", notebook_display=False)
print("Pydeck map saved as 'pydeck_lissajous_domes.html'.")