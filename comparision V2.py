import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymc as pm
import arviz as az
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.stats import multivariate_normal

# =============================================================================
# 1. Data Collection and Preparation
# =============================================================================

# Rocket Data (Estimates - Refine as needed)
rocket_data = {
    "V2": {
        "range_km": 320,
        "tnt_equivalent_kg": 1000,  # 1 ton
        "cd": 0.15,  # Drag coefficient (estimated)
        "speed_km_s": 1.6, # Max speed
        "length_m": 14,
        "diameter_m": 1.65
    },
    "Iskander": {
        "range_km": 500,
        "tnt_equivalent_kg": 480, # Assuming warhead is similar to  Iskander-M
        "cd": 0.25,
        "speed_km_s": 2.1,
        "length_m": 7.3,
        "diameter_m": 0.92
    },
    "Tomahawk": {
        "range_km": 2500,
        "tnt_equivalent_kg": 454,  # 1000 lbs
        "cd": 0.30,
        "speed_km_s": 0.24, # Subsonic
        "length_m": 6.25,
        "diameter_m": 0.52
    }
}

# Convert TNT equivalent to Joules (1 kg TNT ~ 4.184e6 Joules)
for rocket in rocket_data:
    rocket_data[rocket]["energy_joules"] = rocket_data[rocket]["tnt_equivalent_kg"] * 4.184e6

# =============================================================================
# 2. Exponential Curve Comparison
# =============================================================================

fig_exp = make_subplots(rows=3, cols=1, subplot_titles=("Range", "Blast Energy", "Aerodynamics (Drag Coefficient)"))

# Normalize data for plotting
range_norm = [rocket_data[r]["range_km"] / 1000 for r in rocket_data]  # Normalize range
energy_norm = [rocket_data[r]["energy_joules"] / 1e10 for r in rocket_data]  # Normalize energy
cd_norm = [rocket_data[r]["cd"] for r in rocket_data]

rockets = list(rocket_data.keys())

fig_exp.add_trace(go.Bar(x=rockets, y=range_norm, name="Range", marker_color='blue'), row=1, col=1)
fig_exp.add_trace(go.Bar(x=rockets, y=energy_norm, name="Blast Energy", marker_color='green'), row=2, col=1)
fig_exp.add_trace(go.Bar(x=rockets, y=cd_norm, name="Drag Coefficient", marker_color='red'), row=3, col=1)

fig_exp.update_layout(title_text="Exponential Curve Comparison of Rockets", showlegend=False)
fig_exp.show()

# =============================================================================
# 3. 3D Blast Crater Visualization
# =============================================================================
def calculate_crater_radius(energy, material_strength=1e7):
    """ Simplified crater radius calculation (more conceptual than precise)"""
    return (energy / material_strength)**(1/3) #  Volume ~ energy / strength,  r ~ V^(1/3)

# Crater plotting
fig_crater = go.Figure()
for rocket in rockets:
    energy = rocket_data[rocket]["energy_joules"]
    radius = calculate_crater_radius(energy)
    # Create points for a sphere
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    fig_crater.add_trace(go.Surface(x=x, y=y, z=z, name=f"{rocket} Crater", showscale=False))

fig_crater.update_layout(title_text="3D Blast Crater Approximation",
                  scene=dict(aspectmode="cube"))
fig_crater.show()

# =============================================================================
# 4. Pairwise Comparison Plots (Matplotlib)
# =============================================================================

# Create a Pandas DataFrame for easier plotting
df = pd.DataFrame({
    "Rocket": rockets,
    "Range (km)": [rocket_data[r]["range_km"] for r in rockets],
    "Energy (GJ)": [rocket_data[r]["energy_joules"] / 1e9 for r in rockets],  # Convert to GJ for display
    "Drag Coefficient": [rocket_data[r]["cd"] for r in rockets],
    "Speed (km/s)": [rocket_data[r]["speed_km_s"] for r in rockets],
    "Length (m)": [rocket_data[r]["length_m"] for r in rockets],
    "Diameter (m)": [rocket_data[r]["diameter_m"] for r in rockets]
})

# Pairwise scatter plots
fig, axes = plt.subplots(3, 2, figsize=(12, 12))
axes = axes.flatten()
features = ["Range (km)", "Energy (GJ)", "Drag Coefficient", "Speed (km/s)", "Length (m)", "Diameter (m)"]
for i, feature in enumerate(features):
    sns.scatterplot(data=df, x="Rocket", y=feature, ax=axes[i])
    axes[i].set_title(f"Rocket vs {feature}")
plt.tight_layout()
plt.show()

# =============================================================================
# 5. Covariance Matrix
# =============================================================================

# Calculate the covariance matrix
cov_matrix = df[["Range (km)", "Energy (GJ)", "Drag Coefficient", "Speed (km/s)", "Length (m)", "Diameter (m)"]].cov()

# Plot the covariance matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cov_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Covariance Matrix of Rocket Features")
plt.tight_layout()
plt.show()

# =============================================================================
# 6. MCMC Simulation (Illustrative - Range Comparison)
# =============================================================================

#  Simplified MCMC to estimate the mean range of the rockets.  Very basic.

# Prior belief about the range (mean and standard deviation)
prior_mean = np.mean([rocket_data[r]["range_km"] for r in rockets])
prior_std = np.std([rocket_data[r]["range_km"] for r in rockets])

# MCMC model
with pm.Model() as model:
    # Prior for the mean range
    mu = pm.Normal("mu", mu=prior_mean, sigma=prior_std)

    # Likelihood (assuming observed ranges are normally distributed around the true mean)
    observed_ranges = pm.Normal("observed_ranges", mu=mu, sigma=prior_std, observed=[rocket_data[r]["range_km"] for r in rockets])

    # Inference
    idata = pm.sample(1000, tune=1000, chains=2)

# Results visualization
az.plot_posterior(idata, var_names=["mu"], textsize=12)
plt.title("Posterior Distribution of Mean Range")
plt.show()

az.plot_trace(idata, var_names=["mu"], textsize=12)
plt.tight_layout()
plt.show()
