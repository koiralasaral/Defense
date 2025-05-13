import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib import cm
from matplotlib.colors import LightSource

# =============================================================================
# 1. Rocket Database with Enhanced Parameters
# =============================================================================

rocket_data = {
    "V2": {
        "range_km": 320,
        "tnt_equivalent_kg": 1000,
        "cd": 0.15,
        "speed_km_s": 1.6,
        "length_m": 14,
        "diameter_m": 1.65,
        "type": "Ballistic",
        "cost_million_usd": 0.3,
        "energy_joules": 1000 * 4.184e6,
        "mass_kg": 12900,  # Actual V2 mass
        "warhead_mass_kg": 1000
    },
    "Iskander": {
        "range_km": 500,
        "tnt_equivalent_kg": 480,
        "cd": 0.30,
        "speed_km_s": 2.1,
        "length_m": 7.3,
        "diameter_m": 0.92,
        "type": "Ballistic",
        "cost_million_usd": 3,
        "energy_joules": 480 * 4.184e6,
        "mass_kg": 3800,
        "warhead_mass_kg": 480
    },
    "Tomahawk": {
        "range_km": 2500,
        "tnt_equivalent_kg": 454,
        "cd": 0.30,
        "speed_km_s": 0.24,
        "length_m": 6.25,
        "diameter_m": 0.52,
        "type": "Cruise",
        "cost_million_usd": 1.5,
        "energy_joules": 454 * 4.184e6,
        "mass_kg": 1200,
        "warhead_mass_kg": 454
    },
    "Minuteman_III": {
        "range_km": 13000,
        "tnt_equivalent_kg": 300000,
        "cd": 0.20,
        "speed_km_s": 7,
        "length_m": 18.2,
        "diameter_m": 1.67,
        "type": "ICBM",
        "cost_million_usd": 7,
        "energy_joules": 300000 * 4.184e6,
        "mass_kg": 35400,
        "warhead_mass_kg": 300000
    },
    "Topol_M": {
        "range_km": 11000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.22,
        "speed_km_s": 7.3,
        "length_m": 22.7,
        "diameter_m": 1.86,
        "type": "ICBM",
        "cost_million_usd": 8,
        "energy_joules": 1000000 * 4.184e6,
        "mass_kg": 47200,
        "warhead_mass_kg": 1000000
    },
    "DF_41": {
        "range_km": 14000,
        "tnt_equivalent_kg": 1000000,
        "cd": 0.21,
        "speed_km_s": 7.8,
        "length_m": 16.5,
        "diameter_m": 2,
        "type": "ICBM",
        "cost_million_usd": 10,
        "energy_joules": 1000000 * 4.184e6,
        "mass_kg": 52000,
        "warhead_mass_kg": 1000000
    },
    "Sarmat": {
        "range_km": 18000,
        "tnt_equivalent_kg": 8000000,
        "cd": 0.23,
        "speed_km_s": 7.5,
        "length_m": 35.5,
        "diameter_m": 3,
        "type": "ICBM",
        "cost_million_usd": 20,
        "energy_joules": 8000000 * 4.184e6,
        "mass_kg": 208000,
        "warhead_mass_kg": 8000000
    }
}

# Calculate cross-sectional area for all rockets
for name in rocket_data:
    rocket_data[name]['cross_section_m2'] = np.pi * (rocket_data[name]['diameter_m']/2)**2

# =============================================================================
# 2. Atmospheric Models
# =============================================================================

class AtmosphericModel:
    """Base class for atmospheric models"""
    @staticmethod
    def density(altitude):
        raise NotImplementedError
        
    @staticmethod
    def temperature(altitude):
        raise NotImplementedError

class USStandardAtmosphere1976(AtmosphericModel):
    """US Standard Atmosphere 1976 model"""
    @staticmethod
    def density(altitude):
        # Piecewise exponential approximation of US Standard Atmosphere
        # Altitude in km, returns density in kg/m^3
        if altitude < 11:
            return 1.225 * np.exp(-altitude / 8.44)
        elif 11 <= altitude < 25:
            return 0.364 * np.exp(-(altitude - 11) / 6.49)
        elif 25 <= altitude < 47:
            return 0.0186 * (47 - altitude) / 22
        elif 47 <= altitude < 53:
            return 0.0011 * (53 - altitude) / 6
        elif 53 <= altitude < 79:
            return 0.0007 * np.exp(-(altitude - 53) / 7.95)
        elif 79 <= altitude < 90:
            return 0.00004 * (90 - altitude) / 11
        else:
            return 1e-9  # Very low density above 90 km

    @staticmethod
    def temperature(altitude):
        # Temperature in Kelvin
        if altitude < 11:
            return 288.15 - 6.5 * altitude
        elif 11 <= altitude < 20:
            return 216.65
        elif 20 <= altitude < 32:
            return 216.65 + (altitude - 20)
        elif 32 <= altitude < 47:
            return 228.65 + 2.8 * (altitude - 32)
        elif 47 <= altitude < 51:
            return 270.65
        elif 51 <= altitude < 71:
            return 270.65 - 2.8 * (altitude - 51)
        elif 71 <= altitude < 85:
            return 214.65 - 2.0 * (altitude - 71)
        else:
            return 186.65

class ExponentialAtmosphere(AtmosphericModel):
    """Simple exponential atmosphere model"""
    @staticmethod
    def density(altitude):
        # Scale height ~7-8 km for Earth
        return 1.225 * np.exp(-altitude / 7.2)  # kg/m^3

    @staticmethod
    def temperature(altitude):
        return 288.15 - 6.5 * altitude  # Simple linear decrease

# =============================================================================
# 3. Terrain Generation and Visualization
# =============================================================================

class TerrainGenerator:
    """Generates realistic terrain for visualization"""
    @staticmethod
    def generate_terrain(x_range, y_range, resolution=100):
        """Generate terrain with mountains and valleys"""
        x = np.linspace(-x_range, x_range, resolution)
        y = np.linspace(-y_range, y_range, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create multiple terrain features
        Z = np.zeros_like(X)
        
        # Base terrain (hills)
        Z += 0.2 * np.sin(0.5*X) * np.cos(0.3*Y)
        
        # Add some mountains
        Z += 0.5 * np.exp(-(X**2 + (Y-0.5*y_range)**2)/(0.3*x_range)**2)
        Z += 0.3 * np.exp(-((X+0.3*x_range)**2 + (Y+0.2*y_range)**2)/(0.2*x_range)**2)
        
        # Scale to reasonable altitudes (in km)
        Z = Z * 3  
        
        return X, Y, Z
    
    @staticmethod
    def plot_terrain(X, Y, Z, trajectory=None, title="Terrain Map"):
        """Plot terrain with optional trajectory overlay"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Create light source for better visualization
        ls = LightSource(azdeg=315, altdeg=45)
        rgb = ls.shade(Z, cmap=cm.terrain, vert_exag=0.1, blend_mode='soft')
        
        # Plot terrain surface
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                              linewidth=0, antialiased=False, shade=False)
        
        # Plot trajectory if provided
        if trajectory is not None:
            ax.plot(trajectory['x']/1000, trajectory['y']/1000, trajectory['z']/1000,
                    'r-', linewidth=2, label='Trajectory')
            
            # Mark impact point
            impact_x = trajectory['x'][-1]/1000
            impact_y = trajectory['y'][-1]/1000
            impact_z = trajectory['z'][-1]/1000
            ax.scatter([impact_x], [impact_y], [impact_z], c='red', s=100, 
                       marker='*', label='Impact Point')
            
            ax.legend()
        
        ax.set_xlabel('East (km)')
        ax.set_ylabel('North (km)')
        ax.set_zlabel('Altitude (km)')
        ax.set_title(title)
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. Enhanced Projectile Motion with 3D Visualization
# =============================================================================

def enhanced_projectile_motion(rocket_name, launch_angle_deg=45, azimuth_angle_deg=0, 
                             atmosphere_model=USStandardAtmosphere1976, 
                             gravity_model='spherical', dt=0.1, max_time=1000,
                             terrain_range_km=500, terrain_resolution=100):
    """
    Enhanced projectile motion calculation with 3D visualization and sophisticated atmosphere
    
    Args:
        rocket_name: Name of the rocket
        launch_angle_deg: Launch angle from horizontal (degrees)
        azimuth_angle_deg: Launch azimuth angle (degrees) - 0 is due East
        atmosphere_model: Atmospheric model class
        gravity_model: 'spherical' or 'flat'
        dt: Time step (s)
        max_time: Maximum simulation time (s)
        terrain_range_km: Range of terrain to generate (km)
        terrain_resolution: Resolution of terrain grid
        
    Returns:
        Dictionary with trajectory data and visualization
    """
    rocket = rocket_data[rocket_name]
    g0 = 9.80665  # m/s^2 at surface
    R_earth = 6371000  # Earth radius in meters
    
    # Convert angles to radians
    theta = np.radians(launch_angle_deg)
    azimuth = np.radians(azimuth_angle_deg)
    
    # Initial conditions [x, y, z, vx, vy, vz] in meters and m/s
    v0 = rocket['speed_km_s'] * 1000  # Convert km/s to m/s
    initial_state = np.array([
        0,  # x (east)
        0,  # y (north)
        0,  # z (up) - starting at surface
        v0 * np.cos(theta) * np.sin(azimuth),  # vx
        v0 * np.cos(theta) * np.cos(azimuth),  # vy
        v0 * np.sin(theta)  # vz
    ])
    
    def derivatives(t, state):
        x, y, z, vx, vy, vz = state
        r = np.sqrt(x**2 + y**2 + (R_earth + z)**2)
        altitude = r - R_earth
        
        # Gravity magnitude
        if gravity_model == 'spherical':
            g = g0 * (R_earth / r)**2
        else:  # flat
            g = g0
            
        # Gravity components (toward Earth center)
        gx = -g * x / r
        gy = -g * y / r
        gz = -g * (z + R_earth) / r
        
        # Atmospheric density
        rho = atmosphere_model.density(altitude/1000)  # convert m to km
        
        # Velocity magnitude
        v = np.sqrt(vx**2 + vy**2 + vz**2)
        
        # Drag force (Fd = 0.5 * rho * v^2 * Cd * A)
        Fd = 0.5 * rho * v**2 * rocket['cd'] * rocket['cross_section_m2']
        
        # Drag components (opposite velocity)
        if v > 0:
            Fdx = -Fd * vx / v
            Fdy = -Fd * vy / v
            Fdz = -Fd * vz / v
        else:
            Fdx, Fdy, Fdz = 0, 0, 0
            
        # Total acceleration
        ax = Fdx / rocket['mass_kg'] + gx
        ay = Fdy / rocket['mass_kg'] + gy
        az = Fdz / rocket['mass_kg'] + gz
        
        return [vx, vy, vz, ax, ay, az]
    
    # Event for hitting the ground
    def hit_ground(t, state):
        return state[2]  # z coordinate
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    # Solve the ODE
    sol = solve_ivp(derivatives, [0, max_time], initial_state, 
                    events=hit_ground, max_step=dt)
    
    # Extract results
    trajectory = {
        'time': sol.t,
        'x': sol.y[0],
        'y': sol.y[1],
        'z': sol.y[2],
        'vx': sol.y[3],
        'vy': sol.y[4],
        'vz': sol.y[5],
        'rocket': rocket_name,
        'launch_angle': launch_angle_deg,
        'azimuth_angle': azimuth_angle_deg
    }
    
    # Generate terrain
    X, Y, Z = TerrainGenerator.generate_terrain(
        terrain_range_km, terrain_range_km, terrain_resolution)
    
    # Plot 3D trajectory with terrain
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot terrain
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cm.terrain, vert_exag=0.2, blend_mode='soft')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                   linewidth=0, antialiased=False, shade=False, alpha=0.7)
    
    # Plot trajectory
    ax.plot(trajectory['x']/1000, trajectory['y']/1000, trajectory['z']/1000,
            'r-', linewidth=3, label='Trajectory')
    
    # Mark impact point
    impact_x = trajectory['x'][-1]/1000
    impact_y = trajectory['y'][-1]/1000
    impact_z = trajectory['z'][-1]/1000
    ax.scatter([impact_x], [impact_y], [impact_z], c='red', s=200, 
               marker='*', label='Impact Point')
    
    # Add Earth curvature
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    r = 6371  # Earth radius in km
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='blue', alpha=0.1, linewidth=0.5)
    
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Altitude (km)')
    ax.set_title(f"3D Trajectory of {rocket_name} with Terrain\n"
                 f"Launch Angle: {launch_angle_deg}°, Azimuth: {azimuth_angle_deg}°")
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return trajectory

# =============================================================================
# 5. Example Simulations with Visualization
# =============================================================================

# Simulate V2 rocket with different atmospheric models
print("Simulating V2 rocket trajectories...")
v2_traj_exp = enhanced_projectile_motion("V2", launch_angle_deg=45, 
                                       atmosphere_model=ExponentialAtmosphere)
v2_traj_usa = enhanced_projectile_motion("V2", launch_angle_deg=45,
                                       atmosphere_model=USStandardAtmosphere1976)

# Simulate ICBM with terrain visualization
print("\nSimulating ICBM trajectory with terrain...")
icbm_traj = enhanced_projectile_motion("Minuteman_III", launch_angle_deg=50,
                                     azimuth_angle_deg=30, terrain_range_km=1000)

# Simulate cruise missile with low altitude terrain following
print("\nSimulating cruise missile trajectory...")
cruise_traj = enhanced_projectile_motion("Tomahawk", launch_angle_deg=10,
                                        azimuth_angle_deg=45, terrain_range_km=300)

# =============================================================================
# 6. Impact Analysis and Crater Visualization
# =============================================================================

def calculate_impact_effects(trajectory):
    """Calculate impact effects and visualize crater"""
    rocket_name = trajectory['rocket']
    rocket = rocket_data[rocket_name]
    
    # Calculate impact velocity
    v_impact = np.sqrt(trajectory['vx'][-1]**2 + 
                     trajectory['vy'][-1]**2 + 
                     trajectory['vz'][-1]**2)
    
    # Calculate crater dimensions (simplified model)
    # Using energy-based crater scaling (https://impact.ese.ic.ac.uk/ImpactEffects/)
    energy_kt = rocket['energy_joules'] / (4.184e12)  # Convert to kilotons TNT
    crater_diameter_km = 0.1 * energy_kt**(1/3.4)  # Simple scaling law
    crater_depth_km = crater_diameter_km / 5
    
    print(f"\nImpact Analysis for {rocket_name}:")
    print(f"  Impact Velocity: {v_impact/1000:.2f} km/s")
    print(f"  Warhead Yield: {energy_kt:.2f} kt TNT equivalent")
    print(f"  Estimated Crater Diameter: {crater_diameter_km:.2f} km")
    print(f"  Estimated Crater Depth: {crater_depth_km:.2f} km")
    
    # Visualize crater in terrain
    impact_x = trajectory['x'][-1]/1000
    impact_y = trajectory['y'][-1]/1000
    
    # Generate terrain around impact site
    crater_range = crater_diameter_km * 3
    x = np.linspace(impact_x - crater_range, impact_x + crater_range, 100)
    y = np.linspace(impact_y - crater_range, impact_y + crater_range, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create terrain with crater
    R = np.sqrt((X - impact_x)**2 + (Y - impact_y)**2)
    Z = np.zeros_like(X)
    
    # Add crater shape (Gaussian depression)
    Z -= crater_depth_km * np.exp(-(R**2)/(crater_diameter_km/2)**2)
    
    # Add some terrain features
    Z += 0.1 * np.sin(0.5*X) * np.cos(0.3*Y)
    
    # Plot crater
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create light source
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(Z, cmap=cm.terrain, vert_exag=5, blend_mode='soft')
    
    # Plot surface
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=rgb,
                          linewidth=0, antialiased=False, shade=False)
    
    # Mark impact point
    ax.scatter([impact_x], [impact_y], [0], c='red', s=100, 
               marker='*', label='Impact Point')
    
    ax.set_xlabel('East (km)')
    ax.set_ylabel('North (km)')
    ax.set_zlabel('Depth (km)')
    ax.set_title(f"Crater Formation from {rocket_name} Impact\n"
                 f"Diameter: {crater_diameter_km:.2f} km, Depth: {crater_depth_km:.2f} km")
    ax.legend()
    plt.tight_layout()
    plt.show()

# Perform impact analysis
calculate_impact_effects(v2_traj_usa)
calculate_impact_effects(icbm_traj)

# =============================================================================
# 7. Launch Angle Optimization
# =============================================================================

def optimize_launch_angle(rocket_name, atmosphere_model=USStandardAtmosphere1976):
    """Find optimal launch angle for maximum range"""
    angles = np.linspace(20, 70, 20)
    ranges = []
    
    plt.figure(figsize=(12, 6))
    for angle in angles:
        traj = enhanced_projectile_motion(rocket_name, launch_angle_deg=angle,
                                        atmosphere_model=atmosphere_model)
        range_km = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2)/1000
        ranges.append(range_km)
        plt.plot(np.sqrt(traj['x']**2 + traj['y']**2)/1000, traj['z']/1000,
                 alpha=0.6, label=f'{angle}° (Range: {range_km:.1f} km)')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.title(f'{rocket_name} Trajectories at Different Launch Angles')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Plot range vs angle
    plt.figure(figsize=(10, 6))
    plt.plot(angles, ranges, 'o-')
    plt.xlabel('Launch Angle (degrees)')
    plt.ylabel('Range (km)')
    plt.title(f'{rocket_name} Range vs Launch Angle')
    plt.grid(True)
    plt.show()
    
    # Find optimal angle
    optimal_idx = np.argmax(ranges)
    print(f"\nOptimal launch angle for {rocket_name}: {angles[optimal_idx]:.1f}°")
    print(f"Maximum range achieved: {ranges[optimal_idx]:.1f} km")
    print(f"Target range: {rocket_data[rocket_name]['range_km']} km")
    
    return angles[optimal_idx]

# Optimize launch angle for V2
optimal_v2_angle = optimize_launch_angle("V2")

# Optimize launch angle for ICBM
optimal_icbm_angle = optimize_launch_angle("Minuteman_III")

# =============================================================================
# 8. Comprehensive Rocket Comparison
# =============================================================================

def compare_rockets(rocket_names, launch_angle=45):
    """Compare multiple rockets' trajectories"""
    plt.figure(figsize=(12, 8))
    
    for name in rocket_names:
        traj = enhanced_projectile_motion(name, launch_angle_deg=launch_angle)
        range_km = np.sqrt(traj['x'][-1]**2 + traj['y'][-1]**2)/1000
        plt.plot(np.sqrt(traj['x']**2 + traj['y']**2)/1000, traj['z']/1000,
                 label=f'{name} (Range: {range_km:.1f} km)')
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Altitude (km)')
    plt.title(f'Rocket Trajectory Comparison at {launch_angle}° Launch Angle')
    plt.legend()
    plt.grid(True)
    plt.show()

# Compare different rocket types
compare_rockets(["V2", "Tomahawk", "Minuteman_III"])