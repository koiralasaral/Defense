import folium
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpmath import mp, mpf, fac

# Increase the precision in mpmath to mitigate overflow issues
mp.dps = 200  # 200 decimal places


# =============================================================================
# COMMON FUNCTIONS
# =============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Compute the great‐circle distance (in km) between two (lat,lon) points.
    """
    R = 6371  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    """
    Compute a Fourier-perturbed point along the line between start and end.
    Returns (lat, lon).
    """
    lat_lin = start[0] + t * (end[0] - start[0])
    lon_lin = start[1] + t * (end[1] - start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)


def derivative_n_mp(f, t0, n, h):
    """
    Compute the nth derivative of function f at t = t0 using a forward difference formula,
    doing the computation in high precision with mpmath.

    f should return a (lat, lon) tuple.

    Uses the formula:
      f^(n)(t0) ≈ (Σ_(i=0)^n (-1)^(n-i) * binom(n,i) * f(t0 + i*h)) / h^n
    """
    d_lat = mpf('0')
    d_lon = mpf('0')
    for i in range(n + 1):
        coeff = (-1) ** (n - i) * math.comb(n, i)
        t_val = mpf(t0) + i * mpf(h)
        f_val = f(float(t_val))
        d_lat += coeff * mpf(f_val[0])
        d_lon += coeff * mpf(f_val[1])
    return (d_lat / (mpf(h) ** n), d_lon / (mpf(h) ** n))


def choose_h(n):
    """
    Choose an appropriate finite difference step size h based on the derivative order n.
    Typically, for higher n a larger h is needed to reduce cancellation.
    """
    if n <= 10:
        return 1e-5
    elif n <= 30:
        return 1e-4
    elif n <= 100:
        return 1e-3
    else:
        return 1e-1


def taylor_poly_mp(f, t0, N, t, h):
    """
    Compute the Taylor series approximation of f at parameter value t about t0, up to order N.
    Uses high-precision arithmetic for derivative and factorial computation.

    P_N(t) = Σ_(j=0)^N [f^(j)(t0) / j!] * (t - t0)^j
    Returns (lat, lon) as a tuple of floats.
    """
    approx_lat = mpf('0')
    approx_lon = mpf('0')
    dt = mpf(t) - mpf(t0)
    for j in range(N + 1):
        d_j = derivative_n_mp(f, t0, j, h)
        term = (dt ** j) / fac(j)
        approx_lat += d_j[0] * term
        approx_lon += d_j[1] * term
    return (float(approx_lat), float(approx_lon))


# =============================================================================
# PARAMETERS AND FUNCTION DEFINITION
# =============================================================================
# Endpoints for the Fourier curve (battlefront between Washington, DC and Moscow)
start = (38.9072, -77.0369)
end = (55.7558, 37.6173)


# Define our Fourier curve function f
def f_func(t):
    return fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0)


# Choose the expansion point t0 for the Taylor series.
t0 = 0.5

# Define the orders that we want to compute.
# Remove high orders above 20 if desired; here we use orders 1 through 20,
# and then also try up to highest order 30.
max_order = 30  # maximum order for our Taylor approximations
order_list = list(range(1, max_order + 1))

# Sample 101 points on [0,1]
ts = np.linspace(0, 1, 101)

# =============================================================================
# COMPUTE TAYLOR SERIES APPROXIMANTS FOR EACH ORDER (up to max_order)
# =============================================================================
taylor_curves = {}  # Dictionary: order -> list of (lat, lon)
for order in order_list:
    h = choose_h(order)
    curve = [taylor_poly_mp(f_func, t0, order, t, h) for t in ts]
    taylor_curves[order] = curve

# Also compute the original Fourier curve for reference.
fourier_curve_points = [f_func(t) for t in ts]

# =============================================================================
# CREATE A FOLIUM MAP AND ADD THE CURVES AS POLYLINES
# =============================================================================
center_lat = (start[0] + end[0]) / 2
center_lon = (start[1] + end[1]) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
m.add_child(folium.LatLngPopup())
# Add the original Fourier curve (in black).
folium.PolyLine(locations=fourier_curve_points, color="black", weight=4, tooltip="Fourier Curve").add_to(m)

# Use the "jet" colormap to assign a distinct color for each Taylor approximant.
cmap = plt.cm.jet
num_orders = len(order_list)
for idx, order in enumerate(order_list):
    color = mcolors.to_hex(cmap(idx / (num_orders - 1)))
    polyline_points = taylor_curves[order]
    tooltip_text = f"Taylor Series Order {order}"
    folium.PolyLine(locations=polyline_points, color=color, weight=3, tooltip=tooltip_text).add_to(m)

m.save("taylor_fourier_curves_up_to30.html")


print(
    "Folium map with Fourier and Taylor series curves (up to order 30) saved as 'taylor_fourier_curves_up_to30.html'.")

# =============================================================================
# OPTIONAL: MATPLOTLIB PLOT FOR THE CURVES
# =============================================================================
plt.figure(figsize=(10, 6))
fourier_lats = [pt[0] for pt in fourier_curve_points]
fourier_lons = [pt[1] for pt in fourier_curve_points]
plt.plot(fourier_lons, fourier_lats, 'k-', linewidth=3, label="Fourier Curve")
for idx, order in enumerate(order_list):
    color = mcolors.to_hex(cmap(idx / (num_orders - 1)))
    curve = taylor_curves[order]
    lats = [pt[0] for pt in curve]
    lons = [pt[1] for pt in curve]
    plt.plot(lons, lats, color=color, linewidth=2, label=f"Taylor Order {order}")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Fourier Curve and Taylor Series Approximations (Up to Order 30)")
plt.legend(loc='lower right', fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig("taylor_fourier_curves_up_to30.png")
plt.show()
print("Matplotlib plot saved as 'taylor_fourier_curves_up_to30.png'.")