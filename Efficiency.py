import folium
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

# -----------------------------------------------------------------------------
# 1. Deployment Locations in Folium
# -----------------------------------------------------------------------------
# Define illustrative deployment coordinates
deployment_locations = {
    "MIM-104 Patriot": {
        "lat": 38.9072,   # near Washington, DC
        "lon": -77.0369,
        "popup": "US Patriot Deployment (Washington, DC)"
    },
    "Pantsir-S1": {
        "lat": 55.7558,   # near Moscow
        "lon": 37.6173,
        "popup": "Pantsir-S1 Deployment (Moscow)"
    }
}

# Create a basic Folium map with standard markers
m = folium.Map(location=[47.0, 0.0], zoom_start=3)
for system, info in deployment_locations.items():
    folium.Marker(
        [info["lat"], info["lon"]],
        popup=f"<strong>{system}</strong><br>{info['popup']}",
        tooltip=system
    ).add_to(m)

m.save("deployment_map.html")
print("Deployment map saved as 'deployment_map.html'. Open this file in your browser.")

# -----------------------------------------------------------------------------
# 2. Print System Specifications
# -----------------------------------------------------------------------------
# Example specifications gathered from public sources
specifications = {
    "MIM-104 Patriot": {
        "Operational Range": "160 km",
        "Maximum Speed": "5630 km/h",
        "Detection Range": "~130 km",
        "Guidance": "Semi-active radar homing"
    },
    "Pantsir-S1": {
        "Operational Range": "30 km",
        "Maximum Speed": "900 km/h",
        "Detection Range": "~20 km",
        "Guidance": "Active radar and optical tracking"
    }
}
print("\nSystem Specifications:")
for system, specs in specifications.items():
    print(f"\n{system}:")
    for key, value in specs.items():
        print(f"  {key}: {value}")

# -----------------------------------------------------------------------------
# 3. Simulate Performance Data (Operational Range Measurements)
# -----------------------------------------------------------------------------
# We simulate noisy measurements for the operational range.
# For this simulation:
#  - Patriot: true μ = 160 km, sigma = 5 km
#  - Pantsir-S1: true μ = 30 km,  sigma = 2 km
n_perf_samples = 100  # number of performance measurements

true_performance = {
    "MIM-104 Patriot": {"mu": 160, "sigma": 5},
    "Pantsir-S1": {"mu": 30,  "sigma": 2}
}

perf_data = {}  # dictionary to hold simulated data
for system, params in true_performance.items():
    mu_val, sigma_val = params["mu"], params["sigma"]
    # Simulate independent measurements
    perf_data[system] = np.random.normal(mu_val, sigma_val, size=n_perf_samples)

# -----------------------------------------------------------------------------
# 4. MCMC Estimation & Animated Convergence
# -----------------------------------------------------------------------------
# Build a Bayesian model (normal likelihood) for each system, using a vague prior for μ
# and a HalfNormal prior for σ. For a normal model the sample/posterior mean is the complete
# sufficient statistic and hence the MVUE.
mcmc_results = {}
n_mcmc = 2000  # number of MCMC samples
tune_steps = 1000

for system, data_array in perf_data.items():
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=100)          # vague prior for μ
        sigma = pm.HalfNormal("sigma", sigma=10)         # half-normal prior for σ
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=data_array)
        trace = pm.sample(n_mcmc, tune=tune_steps, progressbar=True, random_seed=42, cores=1)
    mcmc_results[system] = trace

# Extract posterior chains and compute the running (cumulative) mean for mu from MCMC samples.
mu_chains = {}
running_mu_means = {}
for system, trace in mcmc_results.items():
    chain = trace.posterior["mu"].values.flatten()  # Extract chain as a 1D array.
    mu_chains[system] = chain
    running_mean = np.cumsum(chain) / np.arange(1, len(chain) + 1)
    running_mu_means[system] = running_mean

# Set up the animated convergence plot for the running means.
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
system_names = list(true_performance.keys())

for ax, system in zip(axes, system_names):
    true_mu = true_performance[system]["mu"]
    ax.axhline(y=true_mu, color='red', linestyle="--", label="True μ")
    ax.set_xlim(0, len(mu_chains[system]))
    ax.set_ylim(true_mu - 10, true_mu + 10)
    ax.set_xlabel("MCMC Iteration")
    ax.set_ylabel("Running Mean of μ")
    ax.set_title(f"{system}")
    ax.legend()

# Initialize a line object for each subplot.
lines = [ax.plot([], [], 'bo-', lw=2)[0] for ax in axes]

def update(frame):
    for i, system in enumerate(system_names):
        xdata = np.arange(1, frame + 1)
        ydata = running_mu_means[system][:frame]
        lines[i].set_data(xdata, ydata)
        axes[i].set_title(f"{system}\nIteration {frame}: μ̂ = {running_mu_means[system][frame-1]:.4f}")
    return lines

anim = FuncAnimation(fig, update, frames=len(mu_chains[system_names[0]]),
                     interval=50, blit=False, repeat=False)

plt.tight_layout()
plt.show()

print("\nMCMC estimation complete. The running mean of μ (posterior mean) estimated from the MCMC samples converges to the true value, demonstrating the MVUE principle under the normal model.")

# -----------------------------------------------------------------------------
# 5. Static Choropleth in Folium Using a Matplotlib Colormap
# -----------------------------------------------------------------------------
# We now use the final performance estimate (last running mean) for each system and
# convert it to a color using the "viridis" colormap.

# Final performance estimates (from the MCMC running mean convergence)
final_estimates = {system: running_mu_means[system][-1] for system in running_mu_means}

# Normalize the performance values across both systems to map to colors.
all_estimates = list(final_estimates.values())
vmin = min(all_estimates)
vmax = max(all_estimates)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.viridis

# Construct a hex color for each system from the normalized performance estimate.
hex_colors = {}
for system, est in final_estimates.items():
    rgba = cmap(norm(est))
    hex_colors[system] = mcolors.to_hex(rgba)

print("\nFinal Performance Estimates with Assigned Colors:")
for system, est in final_estimates.items():
    print(f"  {system}: {est:.2f} km -> Color: {hex_colors[system]}")

# Create a new Folium map with colored CircleMarkers to act as a static choropleth.
m2 = folium.Map(location=[47.0, 0.0], zoom_start=3)
for system, info in deployment_locations.items():
    folium.CircleMarker(
        location=[info["lat"], info["lon"]],
        radius=10,
        popup=f"{system}<br>Estimated Performance: {final_estimates[system]:.2f} km",
        color=hex_colors[system],
        fill=True,
        fill_color=hex_colors[system],
        fill_opacity=0.7
    ).add_to(m2)

m2.save("deployment_performance_choropleth.html")
print("\nDeployment choropleth map saved as 'deployment_performance_choropleth.html'. Open this file in your browser to view the result.")