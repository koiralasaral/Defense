import folium
import numpy as np
from datetime import datetime, timedelta
from folium.plugins import TimestampedGeoJson
import re
from matplotlib.animation import FuncAnimation

# ====================================================
# Helper Functions for Part 2
# ====================================================
def dms_to_dd(dms_str):
    """Converts a DMS string (e.g., "46°37'09.6\"N" or "31°21.8\"E") to decimal degrees."""
    dms_str = dms_str.replace("°", " ").replace("'", " ").replace('"', " ")
    parts = re.split(r"[^\d\w\.]+", dms_str.strip())
    
    try:
        degrees = float(parts[0])
        minutes = float(parts[1])
        direction = parts[-1].upper()
        
        # Check if seconds are provided
        if len(parts) >= 4:
            seconds = float(parts[2])
        else:
            # If no seconds, treat the minutes part as decimal minutes
            minutes = float(parts[1])
            seconds = 0
            
    except (ValueError, IndexError):
        raise ValueError(f"Invalid numeric parts in DMS string: {dms_str}")

    dd = degrees + minutes/60 + seconds/(60*60)
    if direction in ('S', 'W'):
        dd *= -1
    return dd

def weierstrass(t, a_const=0.5, b_const=3, n_terms=50):
    """
    Approximate the Weierstrass function:
      W(t) = sum_{n=0}^{n_terms-1} a_const^n * cos((b_const^n) * np.pi * t)
    """
    return sum((a_const**n) * np.cos((b_const**n) * np.pi * t) for n in range(n_terms))

# ====================================================
# Main Program - Part 2
# ====================================================

# New DMS coordinates to be added
dms_points = [
    "46°22'26.2\"N 31°21.8\"E", "46°35'07.9\"N 24°57.4\"E", "46°33'13.4\"N 32°01.5\"E",
    "46°35'59.5\"N 41°23.3\"E", "46°40'02.2\"N 43°27.1\"E", "46°41'52.5\"N 32°50'38.5\"E", 
    "46°42'23.0\"N 33°00'18.1\"E", "46°46'57.8\"N 15°06.0\"E", "46°50'12.4\"N 33°28'29.3\"E",
    "46°50'15.8\"N 33°35'16.6\"E", "47°02'25.3\"N 33°40'52.8\"E", "47°03'58.8\"N 33°45'14.8\"E",
    "47°05'56.2\"N 33°48'22.1\"E", "47°09'16.7\"N 33°49'43.0\"E", "47°12'43.4\"N 33°54'56.4\"E",
    "47°16'36.2\"N 33°57'45.7\"E", "47°25'22.1\"N 34°10'00.5\"E", "47°24'58.3\"N 34°12'31.2\"E",
    "47°26'42.4\"N 34°18'28.4\"E", "47°30'53.8\"N 23°09.6\"E", "47°30'22.5\"N 26°39.1\"E",
    "47°30'29.5\"N 34°33'40.6\"E", "46°34'09.9\"N 32°35'03.4\"E", "46°34'09.6\"N 32°37'14.3\"E",
    "46°34'59.7\"N 32°37'28.4\"E", "46°21'35.0\"N 30°45'02.6\"E", "46°59'51.1\"N 32°05'51.9\"E",
    "47°25'26.2\"N 34°02'54.1\"E", "47°25'20.9\"N 34°10'08.2\"E", "49°50'18.3\"N 37°34'43.4\"E",
    "49°55'24.2\"N 38°02'14.7\"E", "48°28'44.4\"N 37°54'17.6\"E", "48°35'37.9\"N 37°49'25.7\"E",
    "48°34'49.8\"N 37°50'28.1\"E", "48°34'43.1\"N 37°50'37\"E"
]

# Define a dictionary of strategic points for the map.
# This dictionary will now be populated directly from dms_points.
points = {} 

# Convert DMS points to decimal degrees and add to 'points' dictionary
for idx, dms_str in enumerate(dms_points):
    try:
        # Split DMS string into lat and lon parts
        # The split needs to be robust for cases like "31°21.8\"E" which may not have a space before E
        # A more robust split would be to look for the degree/minute/second symbols or N/S/E/W
        # For this specific format, splitting on the *last* space might be better if the E is always at the end
        # However, the current logic relies on a single space split. Let's make it more robust.
        # Find the last digit/quote before N/S/E/W to separate lat/lon if not cleanly spaced
        match = re.match(r"([\d\W\.]+)([NS])\s*([\d\W\.]+)([EW])", dms_str.strip())
        if match:
            lat_part = match.group(1) + match.group(2)
            lon_part = match.group(3) + match.group(4)
        else:
            # Fallback to splitting on first space if regex doesn't match specific format
            lat_part, lon_part = dms_str.split(" ", 1)

        lat_dd = dms_to_dd(lat_part)
        lon_dd = dms_to_dd(lon_part)
        points[f"Point {idx + 1}"] = (lat_dd, lon_dd)
    except ValueError as e:
        print(f"Error parsing DMS string '{dms_str}': {e}. Skipping this point.")
        continue


# Compute the geographic bounds from the combined set of strategic points.
lats = [coord[0] for coord in points.values()]
lons = [coord[1] for coord in points.values()]
lat_min, lat_max = min(lats), max(lats)
lon_min, lon_max = min(lons), max(lons)

# Compute the center and amplitudes for the Lissajous envelope (main map).
lat_center = (lat_min + lat_max) / 2
lon_center = (lon_min + lon_max) / 2
Ax_full = (lon_max - lon_min) / 2
Ay_full = (lat_max - lat_min) / 2
Ax = 0.8 * Ax_full  # Scaling for a tighter envelope
Ay = 0.8 * Ay_full

# Create the main Folium map with satellite imagery.
m = folium.Map(location=[lat_center, lon_center], zoom_start=6) # Adjusted zoom for wider area
folium.TileLayer('Esri.WorldImagery', name='Satellite').add_to(m)
folium.LayerControl().add_to(m) # Add layer control for satellite imagery

# Add markers for each strategic point.
for name, coords in points.items():
    folium.Marker(location=coords, popup=name, icon=folium.Icon(color='purple')).add_to(m)

# ---------------------------------------------------
# Compute and add the static Lissajous envelope for the main map.
# ---------------------------------------------------
# Parametric equations for the Lissajous envelope:
#   x(t) = lon_center + Ax * np.sin(a_param * t + delta)   [longitude]
#   y(t) = lat_center + Ay * np.sin(b_param * t)           [latitude]
a_param = 3
b_param = 2
delta = np.pi / 2
t_vals = np.linspace(0, 2 * np.pi, 1000) # More points for a smoother curve
lon_curve = lon_center + Ax * np.sin(a_param * t_vals + delta)
lat_curve = lat_center + Ay * np.sin(b_param * t_vals)
# Folium expects (lat, lon) ordering for PolyLine.
lissajous_line = list(zip(lat_curve, lon_curve))
folium.PolyLine(lissajous_line, color='blue', weight=3, opacity=0.7, tooltip='Lissajous Envelope').add_to(m)

# ====================================================
# PART 2: Build a Timestamped GeoJSON Animation Layer
# ====================================================
# For 200 time frames, compute the moving point along the curve.
# For each frame, compute the curvature (for osculating features) and also
# evaluate the Weierstrass function along the trajectory.
features = []
n_frames = 200 # More frames for smoother animation
t_arr = np.linspace(0, 2 * np.pi, n_frames)
base_time = datetime(2025, 5, 25, 14, 36, 0)
weier_scale = 0.05  # Adjusted scale factor for vertical offset (latitude) of Weierstrass function

for i, t_i in enumerate(t_arr):
    # Timestamp for each frame (assume 1-second intervals).
    timestamp = (base_time + timedelta(seconds=i)).isoformat()
    
    # Compute current Lissajous curve point.
    x_i = lon_center + Ax * np.sin(a_param * t_i + delta)  # longitude
    y_i = lat_center + Ay * np.sin(b_param * t_i)           # latitude
    
    # First derivatives.
    dx_dt = Ax * a_param * np.cos(a_param * t_i + delta)
    dy_dt = Ay * b_param * np.cos(b_param * t_i)
    speed = np.hypot(dx_dt, dy_dt)
    
    # Second derivatives.
    d2x_dt2 = -Ax * a_param**2 * np.sin(a_param * t_i + delta)
    d2y_dt2 = -Ay * b_param**2 * np.sin(b_param * t_i)
    
    # Curvature: κ = |x'y'' - y'x''| / (speed³)
    # Add a small epsilon to speed**3 to prevent division by zero for very low speeds
    curvature = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed**3 + 1e-10) # Used 1e-10 for robustness
    
    # --- Feature 1: Moving Lissajous Point ---
    point_feature = {
        "type": "Feature",
        "properties": {
            "times": [timestamp],
            "popup": (f"Time = {t_i:.2f}<br>"
                      f"Curvature Radius = {'inf' if curvature < 1e-8 else f'{1/curvature:.2f} km'}<br>"
                      f"Curvature = {curvature:.5f}"), # More precision
            "icon": "circle",
            "marker-color": "red",
            "iconstyle": {
                "fillColor": "red",
                "fillOpacity": 0.8,
                "radius": 6
            }
        },
        "geometry": {
            "type": "Point",
            "coordinates": [x_i, y_i]  # GeoJSON expects [lon, lat]
        }
    }
    features.append(point_feature)
    
    # --- Feature 2: Osculating Circle and Normal (if curvature is defined and meaningful) ---
    # The radius check (1/curvature) < 5 might be too restrictive given the large map scale.
    # Adjusting threshold or removing if it causes no circles to appear.
    if curvature > 1e-5 and (1/curvature) < 500: # Increased radius threshold for visibility on larger map
        radius = 1 / curvature
        # Unit tangent vector.
        Tx = dx_dt / speed
        Ty = dy_dt / speed
        # Unit normal (rotate tangent by 90° clockwise or counter-clockwise, depending on convention)
        nx = -Ty 
        ny = Tx
        
        # Osculating circle center.
        center_x = x_i + radius * nx
        center_y = y_i + radius * ny
        
        # Sample points along the circle.
        circle_coords = []
        # Adjusting the radius for visualization on map, otherwise circles might be too big/small
        # This scale factor can be tuned based on the map's zoom and the desired visual effect.
        visual_radius_scale = 0.005 # Example scale factor, adjust as needed
        
        for theta in np.linspace(0, 2 * np.pi, 50):
            cx = center_x + radius * visual_radius_scale * np.cos(theta)
            cy = center_y + radius * visual_radius_scale * np.sin(theta)
            circle_coords.append([cx, cy])  # [lon, lat]
        
        circle_feature = {
            "type": "Feature",
            "properties": {
                "times": [timestamp],
                "popup": "Osculating Circle",
                "style": {"color": "lime", "weight": 2, "opacity": 0.7}
            },
            "geometry": {
                "type": "LineString",
                "coordinates": circle_coords
            }
        }
        features.append(circle_feature)
        
        # Normal line feature.
        normal_feature = {
            "type": "Feature",
            "properties": {
                "times": [timestamp],
                "popup": "Normal Line",
                "style": {"color": "cyan", "weight": 2, "dashArray": "5, 5", "opacity": 0.7}
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [x_i, y_i],
                    [center_x + radius * visual_radius_scale * nx, center_y + radius * visual_radius_scale * ny] # Scale normal line end point
                ]
            }
        }
        features.append(normal_feature)
    
    # --- Feature 3: Weierstrass Function Marker along the Trajectory ---
    # Evaluate the Weierstrass function at the current parameter value.
    W_val = weierstrass(t_i)
    # Modify the current point by adding an offset to the latitude.
    # The Weierstrass function introduces a "fractal" perturbation to the latitude.
    x_w = x_i
    y_w = y_i + weier_scale * W_val
    weier_feature = {
        "type": "Feature",
        "properties": {
            "times": [timestamp],
            "popup": f"Weierstrass W(t) = {W_val:.2f}",
            "icon": "star",
            "marker-color": "green",
            "iconstyle": {
                "fillColor": "green",
                "fillOpacity": 0.8,
                "radius": 5
            }
        },
        "geometry": {
            "type": "Point",
            "coordinates": [x_w, y_w]
        }
    }
    features.append(weier_feature)

# Package all features into a GeoJSON FeatureCollection.
geojson_anim = {
    "type": "FeatureCollection",
    "features": features
}

# Create the TimestampedGeoJson layer for animation.
timestamped_layer = TimestampedGeoJson(
    data=geojson_anim,
    period="PT1S",          # 1 second per time frame.
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY/MM/DD HH:mm:ss',
    transition_time=200 # Smooth transition between points
)
m.add_child(timestamped_layer)
m.add_child(folium.LatLngPopup()) # Shows lat/lon on click

# ====================================================
# PART 3: Add a Static Osculating Circle on the Map (for reference)
# ====================================================
# For reference, compute the osculating circle for a mid-animation value.
t_static = t_arr[len(t_arr)//2] # Select a mid-point for static display
x_static = lon_center + Ax * np.sin(a_param * t_static + delta)
y_static = lat_center + Ay * np.sin(b_param * t_static)
dx_dt_static = Ax * a_param * np.cos(a_param * t_static + delta)
dy_dt_static = Ay * b_param * np.cos(b_param * t_static)
speed_static = np.hypot(dx_dt_static, dy_dt_static)
d2x_dt2_static = -Ax * a_param**2 * np.sin(a_param * t_static + delta)
d2y_dt2_static = -Ay * b_param**2 * np.sin(b_param * t_static)
curvature_static = abs(dx_dt_static * d2y_dt2_static - dy_dt_static * d2x_dt2_static) / (speed_static**3 + 1e-10)

if curvature_static > 1e-5 and (1/curvature_static) < 500: # Same meaningfulness check as animation
    radius_static = 1 / curvature_static
    Tx_static = dx_dt_static / speed_static
    Ty_static = dy_dt_static / speed_static
    nx_static = -Ty_static
    ny_static = Tx_static
    center_x_static = x_static + radius_static * nx_static
    center_y_static = y_static + radius_static * ny_static
    
    # Generate points along the static osculating circle.
    circle_coords_static = []
    visual_radius_scale = 0.005 # Example scale factor, adjust as needed
    for theta in np.linspace(0, 2*np.pi, 50):
        cx = center_x_static + radius_static * visual_radius_scale * np.cos(theta)
        cy = center_y_static + radius_static * visual_radius_scale * np.sin(theta)
        # For Polylines, folium expects (lat, lon) ordering.
        circle_coords_static.append((cy, cx))
    
    folium.PolyLine(circle_coords_static, color="orange", weight=2.5, opacity=0.8,
                    tooltip="Static Osculating Circle (mid-animation)").add_to(m)
    folium.CircleMarker(location=[center_y_static, center_x_static], radius=3, color='orange',
                        fill=True, fill_color='orange', fill_opacity=0.8, popup="Osculating Center").add_to(m)


# ====================================================
# PART 4: Add a Static Weierstrass Trajectory Polyline on the Map
# ====================================================
# Here we compute a complete trace of the Weierstrass function along the Lissajous trajectory.
weier_traj = []
for t in t_vals:
    x = lon_center + Ax * np.sin(a_param * t + delta)
    y = lat_center + Ay * np.sin(b_param * t)
    W_val = weierstrass(t)
    y_offset = y + weier_scale * W_val
    # For PolyLine, folium expects points in (lat, lon) ordering.
    weier_traj.append((y_offset, x))
    
folium.PolyLine(weier_traj, color="darkgreen", weight=2.5, opacity=0.8, tooltip="Weierstrass Trajectory").add_to(m)

# ====================================================
# Save the Map to an HTML File
# ====================================================
m.save("map_six_points_with_weierstrass.html")
print("Map with Lissajous envelope, animation (with osculating circle & Weierstrass trace) saved as 'map_six_points_with_weierstrass.html'.")
# Create a matplotlib visualization of the Lissajous curve
import matplotlib.pyplot as plt

# Create figure
plt.figure(figsize=(12, 8))

# Plot the Lissajous curve
plt.plot(lon_curve, lat_curve, 'b-', label='Lissajous Curve')

# Plot the strategic points
plt.scatter([coord[1] for coord in points.values()], 
            [coord[0] for coord in points.values()], 
            c='purple', marker='o', label='Strategic Points')

# Add title and labels
plt.title('Lissajous Curve with Strategic Points')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
# Create an animation of the Lissajous curve with osculating circle
import matplotlib.pyplot as plt

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Animation function
def animate(i):
    ax.clear()
    
    # Plot all strategic points
    ax.scatter([coord[1] for coord in points.values()], 
               [coord[0] for coord in points.values()], 
               c='purple', marker='o', label='Strategic Points')
    
    # Plot full Lissajous curve
    ax.plot(lon_curve, lat_curve, 'b-', alpha=0.3, label='Lissajous Path')
    
    # Current point
    t = t_arr[i]
    x = lon_center + Ax * np.sin(a_param * t + delta)
    y = lat_center + Ay * np.sin(b_param * t)
    
    # Calculate curvature
    dx_dt = Ax * a_param * np.cos(a_param * t + delta)
    dy_dt = Ay * b_param * np.cos(b_param * t)
    speed = np.hypot(dx_dt, dy_dt)
    d2x_dt2 = -Ax * a_param**2 * np.sin(a_param * t + delta)
    d2y_dt2 = -Ay * b_param**2 * np.sin(b_param * t)
    curvature = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed**3 + 1e-10)
    
    # Draw osculating circle if curvature is meaningful
    if curvature > 1e-5 and (1/curvature) < 500:
        radius = 1 / curvature
        Tx = dx_dt / speed
        Ty = dy_dt / speed
        nx = -Ty
        ny = Tx
        center_x = x + radius * nx * 0.005
        center_y = y + radius * ny * 0.005
        
        circle_t = np.linspace(0, 2*np.pi, 50)
        circle_x = center_x + radius * 0.005 * np.cos(circle_t)
        circle_y = center_y + radius * 0.005 * np.sin(circle_t)
        ax.plot(circle_x, circle_y, 'g-', alpha=0.5, label='Osculating Circle')
    
    # Plot current point
    ax.plot(x, y, 'ro', markersize=10, label='Current Position')
    
    ax.set_title('Animated Lissajous Curve')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    
    # Set fixed axis limits based on data bounds
    ax.set_xlim(lon_min - 0.1 * Ax_full, lon_max + 0.1 * Ax_full)
    ax.set_ylim(lat_min - 0.1 * Ay_full, lat_max + 0.1 * Ay_full)

# Create animation
anim = FuncAnimation(fig, animate, frames=len(t_arr), interval=50, repeat=True)
plt.show()
# Create phase-shifted Lissajous curves and plot them
phases = [0, np.pi/2, np.pi, 2*np.pi]
plt.figure(figsize=(15, 10))

for phase in phases:
    # Compute phase-shifted curves
    lon_curve_phase = lon_center + Ax * np.sin(a_param * t_vals + phase)
    lat_curve_phase = lat_center + Ay * np.sin(b_param * t_vals)
    plt.plot(lon_curve_phase, lat_curve_phase, label=f'Phase = {phase:.2f}')

plt.title('Lissajous Curves with Different Phases')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.show()

# Create animation showing phase change
fig, ax = plt.subplots(figsize=(12, 8))
phases_anim = np.linspace(0, 2*np.pi, 100)

def animate_phase(frame):
    ax.clear()
    phase = phases_anim[frame]
    
    # Plot all strategic points
    ax.scatter([coord[1] for coord in points.values()], 
               [coord[0] for coord in points.values()], 
               c='purple', marker='o', label='Strategic Points')
    
    # Plot phase-shifted Lissajous curve
    lon_curve_phase = lon_center + Ax * np.sin(a_param * t_vals + phase)
    lat_curve_phase = lat_center + Ay * np.sin(b_param * t_vals)
    ax.plot(lon_curve_phase, lat_curve_phase, 'b-', label=f'Phase = {phase:.2f}')
    
    ax.set_title('Phase-shifting Lissajous Curve')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(lon_min - 0.1 * Ax_full, lon_max + 0.1 * Ax_full)
    ax.set_ylim(lat_min - 0.1 * Ay_full, lat_max + 0.1 * Ay_full)

anim_phase = FuncAnimation(fig, animate_phase, frames=len(phases_anim), interval=50, repeat=True)
plt.show()

# Create phase-shifted Folium maps
for phase in phases:
    m_phase = folium.Map(location=[lat_center, lon_center], zoom_start=6)
    folium.TileLayer('Esri.WorldImagery', name='Satellite').add_to(m_phase)
    
    # Add markers
    for name, coords in points.items():
        folium.Marker(location=coords, popup=name, icon=folium.Icon(color='purple')).add_to(m_phase)
    
    # Add phase-shifted Lissajous curve
    lon_curve_phase = lon_center + Ax * np.sin(a_param * t_vals + phase)
    lat_curve_phase = lat_center + Ay * np.sin(b_param * t_vals)
    lissajous_line_phase = list(zip(lat_curve_phase, lon_curve_phase))
    folium.PolyLine(lissajous_line_phase, color='blue', weight=3, opacity=0.7, 
                    tooltip=f'Lissajous Curve (Phase={phase:.2f})').add_to(m_phase)
    
    m_phase.save(f'map_phase_{phase:.2f}.html')
    print("The phase-shifted maps have been saved as:")
    for phase in phases:
        print(f"map_phase_{phase:.2f}.html")

        # Enhanced matplotlib visualization with osculating circles, normals, and tangents
        def plot_with_osculating_features(t, ax):
            # Current point on Lissajous curve
            x = lon_center + Ax * np.sin(a_param * t + delta)
            y = lat_center + Ay * np.sin(b_param * t)
            
            # First derivatives
            dx_dt = Ax * a_param * np.cos(a_param * t + delta)
            dy_dt = Ay * b_param * np.cos(b_param * t)
            speed = np.hypot(dx_dt, dy_dt)
            
            # Second derivatives
            d2x_dt2 = -Ax * a_param**2 * np.sin(a_param * t + delta)
            d2y_dt2 = -Ay * b_param**2 * np.sin(b_param * t)
            
            # Curvature
            curvature = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed**3 + 1e-10)
            
            if curvature > 1e-5:
                # Unit tangent vector
                Tx = dx_dt / speed
                Ty = dy_dt / speed
                
                # Unit normal vector
                nx = -Ty
                ny = Tx
                
                # Radius of curvature
                radius = 1/curvature
                
                # Center of osculating circle
                center_x = x + radius * nx
                center_y = y + radius * ny
                
                # Plot osculating circle
                circle_t = np.linspace(0, 2*np.pi, 100)
                circle_x = center_x + radius * np.cos(circle_t)
                circle_y = center_y + radius * np.sin(circle_t)
                ax.plot(circle_x, circle_y, 'g--', alpha=0.3)
                
                # Plot normal vector
                scale = 0.5  # Adjust scale for visualization
                ax.arrow(x, y, nx*scale, ny*scale, color='red', width=0.02)
                
                # Plot tangent vector
                ax.arrow(x, y, Tx*scale, Ty*scale, color='blue', width=0.02)

        # Create animation with osculating features
        fig, ax = plt.subplots(figsize=(12, 8))

        def animate_with_features(i):
            ax.clear()
            t = t_arr[i]
            
            # Plot full Lissajous curve
            ax.plot(lon_curve, lat_curve, 'k-', alpha=0.3)
            
            # Plot strategic points
            ax.scatter([coord[1] for coord in points.values()], 
                       [coord[0] for coord in points.values()],
                       c='purple', marker='o')
            
            # Add osculating features
            plot_with_osculating_features(t, ax)
            
            # Current point
            x = lon_center + Ax * np.sin(a_param * t + delta)
            y = lat_center + Ay * np.sin(b_param * t)
            ax.plot(x, y, 'ro')
            
            ax.set_title('Lissajous Curve with Osculating Features')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True)
            ax.set_xlim(lon_min - 0.1 * Ax_full, lon_max + 0.1 * Ax_full)
            ax.set_ylim(lat_min - 0.1 * Ay_full, lat_max + 0.1 * Ay_full)

        anim_features = FuncAnimation(fig, animate_with_features, frames=len(t_arr), 
                                    interval=50, repeat=True)
        plt.show()
        # Create polar coordinate visualization of the Lissajous curve
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': 'polar'})

        def animate_polar(frame):
            ax.clear()
            t = t_arr[frame]
            
            # Convert all points to polar coordinates
            r_points = []
            theta_points = []
            for coord in points.values():
                # Convert from lat/lon to relative cartesian coordinates
                x = coord[1] - lon_center
                y = coord[0] - lat_center
                # Convert to polar
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                r_points.append(r)
                theta_points.append(theta)
            
            # Convert Lissajous curve to polar coordinates
            x_curve = lon_curve - lon_center
            y_curve = lat_curve - lat_center
            r_curve = np.sqrt(x_curve**2 + y_curve**2)
            theta_curve = np.arctan2(y_curve, x_curve)
            
            # Current point
            x = Ax * np.sin(a_param * t + delta)
            y = Ay * np.sin(b_param * t)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Plot
            ax.plot(theta_curve, r_curve, 'b-', alpha=0.3, label='Lissajous Path')
            ax.scatter(theta_points, r_points, c='purple', marker='o', label='Strategic Points')
            ax.plot(theta, r, 'ro', markersize=10, label='Current Position')
            
            ax.set_title('Polar Lissajous Curve')
            ax.legend()
            ax.grid(True)

        # Create and display the animation
        anim_polar = FuncAnimation(fig, animate_polar, frames=len(t_arr), interval=50, repeat=True)
        plt.show()