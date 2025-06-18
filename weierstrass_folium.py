import folium
import numpy as np
from datetime import datetime, timedelta
from folium.plugins import TimestampedGeoJson
import re
import folium
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# ====================================================
# Helper Functions
# ====================================================
# Here is the list of DMS points to be used in the map.
dms_points = ["46°22'26.2\"N 31°21.8\"E", "46°35'07.9\"N 24°57.4\"E", "46°33'13.4\"N 32°01.5\"E",
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
import re
# Function to convert DMS (Degrees, Minutes, Seconds) to Decimal Degrees (DD)
# This function takes a DMS string and converts it to decimal degrees.

def dms_to_dd(dms_str):
    """Converts a DMS string to decimal degrees."""
    dms_str = dms_str.replace("°", " ").replace("'", " ").replace('"', " ")
    parts = re.split(r"[^\d\w\.]+", dms_str.strip())
    degrees, minutes, seconds, direction = parts[0], parts[1], parts[2], parts[3]
    dd = float(degrees) + float(minutes)/60 + float(seconds)/(60*60)
    if direction in ('S', 'W'):
        dd *= -1
    return dd



def convert_dms_points(dms_points):         
    """Converts a list of DMS coordinate strings to lat/lon pairs."""       
    points = {}     
    for i, point in enumerate(dms_points, 1):         
        lat_str, lon_str = point.split()        
        lat = dms_to_dd(lat_str)        
        lon = dms_to_dd(lon_str)        
        points[f"Point {i}"] = (lat, lon)       
    return points

coord1_str = dms_points[0]     
coord2_str = dms_points[1]

points = convert_dms_points(dms_points)
def interpolate_coords(coord1_str, coord2_str):         
  """Interpolates between two coordinates represented as DMS strings."""            
  lat1_str, lon1_str = coord1_str.split()           
  lat2_str, lon2_str = coord2_str.split()            
  lat1 = dms_to_dd(lat1_str)             
  lon1 = dms_to_dd(lon1_str)            
  lat2 = dms_to_dd(lat2_str)            
  lon2 = dms_to_dd(lon2_str)            
  lat_interpolated = (lat1 + lat2) / 2           
  lon_interpolated = (lon1 + lon2) / 2           
  return lat_interpolated, lon_interpolated          

def taylor_polynomial_regression(points, degree=1):
    """
    Calculates the Taylor polynomial using linear regression.
    Args:
        points (list of tuples): List of (x, y) coordinates.
        degree (int): Degree of the Taylor polynomial (default is 1 for linear).
    Returns:
        LinearRegression: Fitted linear regression model.
    """
    X = np.array([[x**i for i in range(degree + 1)] for x, _ in points])
    y = np.array([y for _, y in points])
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def lissajous_curve(t, A, B, a, b, delta):
    """
    Calculates the x and y coordinates of a Lissajous curve.
    Args:
        t (float or array): Parameter.
        A (float): Amplitude of x.
        B (float): Amplitude of y.
        a (float): Frequency of x.
        b (float): Frequency of y.
        delta (float): Phase difference.
    Returns:
        tuple: (x, y) coordinates.
    """
    x = A * np.sin(a * t + delta)
    y = B * np.sin(b * t)
    return x, y

# ====================================================
# Main Program
# ====================================================

# 1. Coordinate Interpolation
# Add additional points to the map


lat_interpolated, lon_interpolated = interpolate_coords(coord1_str, coord2_str)
print(f"Interpolated Coordinates: Latitude = {lat_interpolated}, Longitude = {lon_interpolated}")

# 2. Folium Map
m = folium.Map(location=[lat_interpolated, lon_interpolated], zoom_start=10)
folium.Marker([lat_interpolated, lon_interpolated], popup="Interpolated Point").add_to(m)

# 3. Taylor Polynomial (Linear Regression)

model = taylor_polynomial_regression(points)
print(f"Taylor Polynomial Coefficients (Linear): Intercept = {model.intercept_}, Slope = {model.coef_[1] if len(model.coef_) > 1 else 0}")
# Interpolate between each consecutive pair of points and add to map
interpolated_coords = []
for i in range(len(dms_points) - 1):
    coord1 = dms_points[i]
    coord2 = dms_points[i + 1]
    lat_interp, lon_interp = interpolate_coords(coord1, coord2)
    interpolated_coords.append((lat_interp, lon_interp))
    # Add to folium map
    folium.CircleMarker(
        location=(lat_interp, lon_interp),
        radius=4,
        color='purple',
        fill=True,
        fill_color='purple',
        popup=f'Interpolated {i+1}',
        tooltip=f'Interpolated {i+1}'
    ).add_to(m)

# 4. Lissajous Curve
# Example parameters
A = 1  # Amplitude of x
B = 1  # Amplitude of y
a = 2  # Frequency of x
b = 3  # Frequency of y
delta = np.pi / 2  # Phase difference

t_vals = np.linspace(0, 2 * np.pi, 200)
lissajous_points = [lissajous_curve(t, A, B, a, b, delta) for t in t_vals]

# Scale and offset Lissajous points to map coordinates (example)
lon_center = lon_interpolated
lat_center = lat_interpolated
lon_scale = 0.1
lat_scale = 0.1

scaled_lissajous_points = [
    (lat_center + lat_scale * y, lon_center + lon_scale * x) for x, y in lissajous_points
]

folium.PolyLine(scaled_lissajous_points, color="red", tooltip="Lissajous Curve").add_to(m)

# 5. Save the Map
m.save("map_with_interpolation_lissajous.html")
print("Map saved as 'map_with_interpolation_lissajous.html'")
# ====================================================
# Helper: Define the Weierstrass Function
# ====================================================
def weierstrass(t, a_const=0.5, b_const=3, n_terms=50):
    """
    Approximate the Weierstrass function:
      W(t) = sum_{n=0}^{n_terms-1} a_const^n * cos((b_const^n) * np.pi * t)
    """
    return sum((a_const**n) * np.cos((b_const**n) * np.pi * t) for n in range(n_terms))

# ====================================================
# PART 1: Create a Folium Map with Satellite Imagery & Markers
# ====================================================


# Compute the geographic bounds.
lats = [coord[0] for coord in points.values()]
lons = [coord[1] for coord in points.values()]
lat_min, lat_max = min(lats), max(lats)
lon_min, lon_max = min(lons), max(lons)

# Compute the center and amplitudes for the Lissajous envelope.
lat_center = (lat_min + lat_max) / 2
lon_center = (lon_min + lon_max) / 2
Ax_full = (lon_max - lon_min) / 2
Ay_full = (lat_max - lat_min) / 2
Ax = 0.8 * Ax_full  # scaling for a tighter envelope
Ay = 0.8 * Ay_full

# Create the Folium map.
m = folium.Map(location=[lat_center, lon_center], zoom_start=5)
folium.TileLayer('Esri.WorldImagery', name='Satellite').add_to(m)

# Add markers for each strategic point.
for name, coords in points.items():
    folium.Marker(location=coords, popup=name).add_to(m)

# ---------------------------------------------------
# Compute and add the static Lissajous envelope.
# ---------------------------------------------------
# Parametric equations:
#   x(t) = lon_center + Ax * sin(a*t + delta)   [longitude]
#   y(t) = lat_center + Ay * sin(b*t)             [latitude]
a_param = 3
b_param = 2
delta = np.pi / 2
t_vals = np.linspace(0, 2 * np.pi, 1000)
lon_curve = lon_center + Ax * np.sin(a_param * t_vals + delta)
lat_curve = lat_center + Ay * np.sin(b_param * t_vals)
# Folium expects (lat, lon) ordering for PolyLine.
lissajous_line = list(zip(lat_curve, lon_curve))
folium.PolyLine(lissajous_line, color='blue', weight=2.5, tooltip='Lissajous Envelope').add_to(m)

# ====================================================
# PART 2: Build a Timestamped GeoJSON Animation Layer
# ====================================================
# For 100 time frames, compute the moving point along the curve.
# For each frame, compute the curvature (for osculating features) and also
# evaluate the Weierstrass function along the trajectory.
features = []
n_frames = 100
t_arr = np.linspace(0, 2 * np.pi, n_frames)
base_time = datetime(2025, 5, 25, 14, 36, 0)
weier_scale = 0.2  # Scale factor for vertical offset (latitude) of Weierstrass function

for i, t_i in enumerate(t_arr):
    # Timestamp for each frame (assume 1-second intervals).
    timestamp = (base_time + timedelta(seconds=i)).isoformat()
    
    # Compute current Lissajous curve point.
    x_i = lon_center + Ax * np.sin(a_param * t_i + delta)  # longitude
    y_i = lat_center + Ay * np.sin(b_param * t_i)            # latitude
    
    # First derivatives.
    dx_dt = Ax * a_param * np.cos(a_param * t_i + delta)
    dy_dt = Ay * b_param * np.cos(b_param * t_i)
    speed = np.hypot(dx_dt, dy_dt)
    # Second derivatives.
    d2x_dt2 = -Ax * a_param**2 * np.sin(a_param * t_i + delta)
    d2y_dt2 = -Ay * b_param**2 * np.sin(b_param * t_i)
    
    # Curvature: κ = |x'y'' - y'x''| / (speed³)
    curvature = abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / (speed**3 + 1e-8)
    
    # --- Feature 1: Moving Lissajous Point ---
    point_feature = {
        "type": "Feature",
        "properties": {
            "times": [timestamp],
            "popup": (f"t = {t_i:.2f}<br>"
                      f"Radius = {'inf' if curvature < 1e-8 else f'{1/curvature:.2f}'}<br>"
                      f"Curvature = {curvature:.3f}"),
            "icon": "circle",
            "marker-color": "red"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [x_i, y_i]  # GeoJSON expects [lon, lat]
        }
    }
    features.append(point_feature)
    
    # --- Feature 2: Osculating Circle and Normal (if curvature is defined) ---
    if curvature >= 1e-8 and (1/curvature) < 1e5:
        radius = 1 / curvature
        # Unit tangent vector.
        Tx = dx_dt / speed
        Ty = dy_dt / speed
        # Unit normal (rotate tangent by 90°).
        nx = -Ty
        ny = Tx
        # Osculating circle center.
        center_x = x_i + radius * nx
        center_y = y_i + radius * ny
        
        # Sample points along the circle.
        circle_coords = []
        for theta in np.linspace(0, 2 * np.pi, 50):
            cx = center_x + radius * np.cos(theta)
            cy = center_y + radius * np.sin(theta)
            circle_coords.append([cx, cy])  # [lon, lat]
        
        circle_feature = {
            "type": "Feature",
            "properties": {
                "times": [timestamp],
                "popup": "Osculating Circle",
                "style": {"color": "blue", "weight": 2}
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
                "style": {"color": "magenta", "weight": 2, "dashArray": "5, 5"}
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [x_i, y_i],
                    [center_x, center_y]
                ]
            }
        }
        features.append(normal_feature)
    
    # --- Feature 3: Weierstrass Function Marker along the Trajectory ---
    # Evaluate the Weierstrass function at the current parameter value.
    W_val = weierstrass(t_i)
    # Modify the current point by adding an offset to the latitude.
    x_w = x_i
    y_w = y_i + weier_scale * W_val
    weier_feature = {
        "type": "Feature",
        "properties": {
            "times": [timestamp],
            "popup": f"Weierstrass W(t) = {W_val:.2f}",
            "icon": "star",
            "marker-color": "green"
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

# Create the TimestampedGeoJson layer.
timestamped_layer = TimestampedGeoJson(
    data=geojson_anim,
    period="PT1S",         # 1 second per time frame.
    add_last_point=True,
    auto_play=True,
    loop=True,
    max_speed=1,
    loop_button=True,
    date_options='YYYY/MM/DD HH:mm:ss'
)
m.add_child(timestamped_layer)
m.add_child(folium.LatLngPopup())

# ====================================================
# PART 3: Add a Static Osculating Circle on the Map
# ====================================================
# For reference, compute the osculating circle for a mid-animation value.
t_static = t_arr[len(t_arr)//2]
x_static = lon_center + Ax * np.sin(a_param * t_static + delta)
y_static = lat_center + Ay * np.sin(b_param * t_static)
dx_dt_static = Ax * a_param * np.cos(a_param * t_static + delta)
dy_dt_static = Ay * b_param * np.cos(b_param * t_static)
speed_static = np.hypot(dx_dt_static, dy_dt_static)
d2x_dt2_static = -Ax * a_param**2 * np.sin(a_param * t_static + delta)
d2y_dt2_static = -Ay * b_param**2 * np.sin(b_param * t_static)
curvature_static = abs(dx_dt_static * d2y_dt2_static - dy_dt_static * d2x_dt2_static) / (speed_static**3 + 1e-8)

if curvature_static >= 1e-8 and (1/curvature_static) < 1e5:
    radius_static = 1 / curvature_static
    Tx_static = dx_dt_static / speed_static
    Ty_static = dy_dt_static / speed_static
    nx_static = -Ty_static
    ny_static = Tx_static
    center_x_static = x_static + radius_static * nx_static
    center_y_static = y_static + radius_static * ny_static
    
    # Generate points along the static osculating circle.
    circle_coords_static = []
    for theta in np.linspace(0, 2*np.pi, 50):
        cx = center_x_static + radius_static * np.cos(theta)
        cy = center_y_static + radius_static * np.sin(theta)
        # For Polylines, folium expects (lat, lon) ordering.
        circle_coords_static.append((cy, cx))
    
    folium.PolyLine(circle_coords_static, color="orange", weight=2.5,
                    tooltip="Static Osculating Circle (mid-animation)").add_to(m)

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
    
folium.PolyLine(weier_traj, color="green", weight=2.5, tooltip="Weierstrass Trajectory").add_to(m)

# ====================================================
# Save the Map to an HTML File
# ====================================================
m.save("map_six_points_with_weierstrass.html")
print("Map with Lissajous envelope, animation "
      "(with osculating circle & Weierstrass trace) "
      "saved as 'map_six_points_with_weierstrass.html'.")