import folium
from folium.plugins import TimestampedGeoJson
import numpy as np
import math
from mpmath import mp, mpf, fac
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pymc as pm
import arviz as az
import datetime
from multiprocessing import freeze_support

# ----------------------------------------------------------------------------
# Real system specifications (with citations in comments)
# ----------------------------------------------------------------------------
# MIM-104 Patriot: range=160 km, cost=4e6 USD/missile, mass=900 kg
patriot_spec = {"range_km": 160, "cost_usd": 4e6, "mass_kg": 900}
# Pantsir-S1: range=20 km, cost=13.5e6 USD/unit, mass=150 kg
pantsir_spec = {"range_km": 20, "cost_usd": 13.5e6, "mass_kg": 150}

def fourier_curve(t, start, end, A=2.0, B=3.0, k=3, phase=0):
    lat_lin = start[0] + t*(end[0]-start[0])
    lon_lin = start[1] + t*(end[1]-start[1])
    lat_pert = A * np.sin(2 * np.pi * k * t + phase)
    lon_pert = B * np.cos(2 * np.pi * k * t + phase)
    return (lat_lin + lat_pert, lon_lin + lon_pert)

# high-precision Taylor utilities
mp.dps = 200

def derivative_n_mp(f, t0, n, h):
    dlat = mpf('0'); dlon = mpf('0')
    for i in range(n+1):
        coeff = (-1)**(n-i) * math.comb(n, i)
        t_val = mpf(t0) + i*mpf(h)
        lat, lon = f(float(t_val))
        dlat += coeff * mpf(lat)
        dlon += coeff * mpf(lon)
    return (dlat/(mpf(h)**n), dlon/(mpf(h)**n))

def taylor_poly_mp(f, t0, N, t, h):
    approx_lat = mpf('0'); approx_lon = mpf('0')
    dt = mpf(t) - mpf(t0)
    for j in range(N+1):
        dlat, dlon = derivative_n_mp(f, t0, j, h)
        term = (dt**j) / fac(j)
        approx_lat += dlat * term
        approx_lon += dlon * term
    return (float(approx_lat), float(approx_lon))

def choose_h(n):
    if n <= 10:   return 1e-5
    if n <= 100:  return 1e-3
    if n <= 1000: return 1e-1
    return 1.0

# RK4 trajectory with drag
g = 9.81  # m/s^2
def rk4_step(state, dt, k, m):
    # state = [x, y, vx, vy]
    def accel(vx, vy):
        v = math.hypot(vx, vy)
        ax = -k * v * vx / m
        ay = -g - (k * v * vy / m)
        return ax, ay

    x, y, vx, vy = state
    ax1, ay1 = accel(vx, vy)
    vx1, vy1 = vx + ax1*dt/2, vy + ay1*dt/2
    ax2, ay2 = accel(vx1, vy1)
    vx2, vy2 = vx + ax2*dt/2, vy + ay2*dt/2
    ax3, ay3 = accel(vx2, vy2)
    vx3, vy3 = vx + ax3*dt, vy + ay3*dt
    ax4, ay4 = accel(vx3, vy3)

    x_new  = x + dt*(vx + 2*vx1 + 2*vx2 + vx3)/6
    y_new  = y + dt*(vy + 2*vy1 + 2*vy2 + vy3)/6
    vx_new = vx + dt*(ax1 + 2*ax2 + 2*ax3 + ax4)/6
    vy_new = vy + dt*(ay1 + 2*ay2 + 2*ay3 + ay4)/6
    return [x_new, y_new, vx_new, vy_new]

# simulate trajectories
def simulate_trajectory(v0, theta, k, m, dt=0.01):
    theta_r = math.radians(theta)
    state = [0, 0, v0*math.cos(theta_r), v0*math.sin(theta_r)]
    traj = [state.copy()]
    while state[1] >= 0:
        state = rk4_step(state, dt, k, m)
        traj.append(state.copy())
    return np.array(traj)

def simulate_weapons_at_point(n, spec):
    v0s = np.random.uniform(150, 300, size=n)
    thetas = np.random.uniform(20, 80, size=n)
    k = 0.1  # drag coeff
    m = spec['mass_kg']
    results = []
    for v0, θ in zip(v0s, thetas):
        traj = simulate_trajectory(v0, θ, k, m)
        energy = 0.5 * m * v0**2
        results.append({"v0":v0, "theta":θ, "energy":energy, "traj":traj})
    return sorted(results, key=lambda x: x['energy'], reverse=True)

# main workflow
def main():
    start = (38.9072, -77.0369)
    end   = (55.7558,  37.6173)
    n_points = 20
    t_values = np.linspace(0, 1, n_points)
    curve_points = [fourier_curve(t, start, end) for t in t_values]
    center = [(start[0]+end[0])/2, (start[1]+end[1])/2]

    # 2D Folium map
    m2d = folium.Map(location=center, zoom_start=4)
    folium.PolyLine(locations=curve_points, color="black", weight=4).add_to(m2d)
    # Taylor overlays
    order, window, samples = 10, 0.1, 30
    cmap = plt.cm.plasma
    for idx, t0 in enumerate(t_values):
        ts_local = np.linspace(max(0,t0-window), min(1,t0+window), samples)
        pts = [taylor_poly_mp(lambda u: fourier_curve(u,start,end), t0, order, t, choose_h(order)) for t in ts_local]
        folium.PolyLine(locations=pts, color=mcolors.to_hex(cmap(idx/(n_points-1))), weight=2).add_to(m2d)
    # trajectories & markers
    for pt in curve_points:
        for spec, col in [(patriot_spec, 'blue'), (pantsir_spec, 'red')]:
            sims = simulate_weapons_at_point(10, spec)
            for w in sims:
                traj = w['traj']
                coords = [[pt[0]+p[1]/1000, pt[1]+p[0]/1000] for p in traj]
                folium.PolyLine(locations=coords, color=col, weight=1, opacity=0.6).add_to(m2d)
            folium.CircleMarker(location=pt, radius=3 + sims[0]['energy']/1e5,
                                color=col, fill=True).add_to(m2d)
    m2d.save("battlefront_weapons_2d.html")

    # 3D animation
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    all_sims = [simulate_weapons_at_point(10, patriot_spec) + simulate_weapons_at_point(10, pantsir_spec) for _ in curve_points]
    lines = [ax.plot([],[],[])[0] for _ in range(len(curve_points)*20)]
    min_lat = min(p[0] for p in curve_points)
    max_lat = max(p[0] for p in curve_points)
    min_lon = min(p[1] for p in curve_points)
    max_lon = max(p[1] for p in curve_points)
    max_alt = max(w['traj'][:,1].max() for sims in all_sims for w in sims)

    def init():
        ax.set_xlim(min_lon, max_lon)
        ax.set_ylim(min_lat, max_lat)
        ax.set_zlim(0, max_alt)
        return lines

    def update(frame):
        pt_idx = frame // 20
        traj_idx = frame % 20
        sims = all_sims[pt_idx]
        w = sims[traj_idx]
        traj = w['traj']
        lons = traj[:,0]/1000 + curve_points[pt_idx][1]
        lats = traj[:,1]/1000 + curve_points[pt_idx][0]
        alts = traj[:,1]
        line = lines[frame]
        line.set_data(lons[:frame], lats[:frame])
        line.set_3d_properties(alts[:frame])
        return [line]

    ani3d = FuncAnimation(fig, update, frames=len(lines), init_func=init, blit=True)
    ani3d.save("trajectories_3d.gif", fps=20)
    # time-stamped animation
    features = []
    t0 = datetime.datetime.utcnow()
    for i, pt in enumerate(curve_points):
        land = simulate_weapons_at_point(10, patriot_spec)[0]['traj'][-1]
        features.append({
            "type":"Feature",
            "geometry":{"type":"Point","coordinates":[pt[1]+land[0]/1000, pt[0]+land[1]/1000]},
            "properties":{"time":(t0+datetime.timedelta(seconds=5*i)).isoformat(),
                            "popup":f"E={land[0]:.0f}"}
        })
    geo = {"type":"FeatureCollection","features":features}
    m_time = folium.Map(location=center, zoom_start=4)
    TimestampedGeoJson(geo, period="PT5S", add_last_point=True,
                       auto_play=True, loop=False).add_to(m_time)
    m_time.save("battlefront_weapons_timed.html")

if __name__ == '__main__':
    freeze_support()
    main()
