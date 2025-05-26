import pydeck as pdk
import json

# GeoJSON data for Nepal (simplified outline for demonstration)
# In a real application, you would load a more detailed GeoJSON file.
# For simplicity, I'm providing a very basic representation.
nepal_geojson = {
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [80.0, 27.0],
            [88.0, 27.0],
            [88.0, 30.0],
            [80.0, 30.0],
            [80.0, 27.0]
        ]]
    },
    "properties": {
        "name": "Nepal"
    }
}

# The centroid of Nepal is approximately 84.19°E, 28.39°N
# You can get more precise coordinates for a better view.
VIEW_STATE = pdk.ViewState(
    latitude=28.39,
    longitude=84.19,
    zoom=6,
    pitch=0
)

# Create a SolidPolygonLayer for Nepal
polygon_layer = pdk.Layer(
    "SolidPolygonLayer",
    data=[nepal_geojson['geometry']['coordinates']], # Use the coordinates from the GeoJSON
    get_polygon="@@0",  # Access the polygon coordinates
    filled=True,
    get_fill_color=[255, 0, 0, 160],  # Red color, 160 opacity
    stroked=True,
    get_line_color=[0, 0, 0],
    get_line_width=100, # Adjust for visibility
    line_width_min_pixels=1
)

# Create the pydeck map
r = pdk.Deck(
    layers=[polygon_layer],
    initial_view_state=VIEW_STATE,
    tooltip={"text": "{name}"}
)

# To render in a Jupyter Notebook, simply call r
# r

# To save as an HTML file (useful if not in a Jupyter environment)
r.to_html("nepal_map.html")
print("Map saved to nepal_map.html")