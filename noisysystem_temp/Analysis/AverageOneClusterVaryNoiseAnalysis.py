import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from particle.processing import get_master_yaml, match_parameters, load_traj_data
from particle.statistics import calculate_avg_vel, calculate_l1_convergence

# Standard plotting choices
# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    # "D": 0.05,
    "phi": "Gamma",
    "gamma": 0.05,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_normal_dn",
    "T_end": 2000.0,
    # "dt": 0.015,
}

os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

final_plot_time = 5000000

yaml_path = (
    "./Experiments/one_cluster_vary_noise_scale_dt_100_runs_larger_gamma_long_run"
)
data_path = "./Experiments/Data.nosync/"

history = get_master_yaml(yaml_path)


fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 5), sharex=True)

# Create colour bar and scale
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0, vcenter=0.25, vmax=0.5)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
# Set tick locations
cbar = fig.colorbar(scalarMap, ticks=np.arange(0, 0.5, 0.05))


# For each matching desired parameters, calculate the l1 error and plot

for diffusion in np.arange(0.05, 0.5, 0.05).tolist():
    search_parameters["D"] = diffusion
    file_names = match_parameters(search_parameters, history)
    first_iter = True
    for idx, file_name in enumerate(file_names):
        simulation_parameters = history[file_name]
        t, x, v = load_traj_data(file_name)
        error = calculate_l1_convergence(t, x, v, final_plot_time=final_plot_time)
        avg_vel = calculate_avg_vel(t, x, v)
        if first_iter:
            avg_vel_store = np.zeros((len(file_names), len(avg_vel)))
            error_store = np.zeros((len(file_names), len(error)))
            first_iter = False

        avg_vel_store[idx, :] = avg_vel
        error_store[idx, :] = error
        # ax1.semilogx(
        #     t,
        #     error,
        #     color=scalarMap.to_rgba(simulation_parameters["D"]),
        #     label=f"{simulation_parameters['D']}",
        #     alpha=0.01,
        #     zorder=1,
        # )
        # ax2.semilogx(
        #     t,
        #     avg_vel,
        #     color=scalarMap.to_rgba(simulation_parameters["D"]),
        #     label=f"{simulation_parameters['D']}",
        #     alpha=0.01,
        #     zorder=1,
        # )

    ax1.semilogx(
        t,
        error_store.mean(axis=0),
        color=scalarMap.to_rgba(simulation_parameters["D"]),
        label=f"{simulation_parameters['D']}",
        alpha=0.8,
        zorder=2,
    )
    ax2.semilogx(
        t,
        avg_vel_store.mean(axis=0),
        color=scalarMap.to_rgba(simulation_parameters["D"]),
        label=f"{simulation_parameters['D']}",
        alpha=0.8,
        zorder=2,
    )
    # if simulation_parameters["D"] == 0.05:
    #     _t, _x, _v = load_traj_data(file_name, simulation_parameters, data_path)
ax1.plot([0, t[-1]], [7.5, 7.5], "k--", alpha=0.2)
ax1.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
ax2.set(xlabel="Time", ylabel=r"$M^N(t)$")
cbar.set_label(r"Diffusion $\sigma$", rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.9, wspace=0.23)
plt.tight_layout()
# plt.show()
# ani = anim_torus(_t, _x, _v, subsample=50)
plt.show()

# fig.savefig(f"OneClusterVaryNoiselargerGamma.jpg", dpi=300)
