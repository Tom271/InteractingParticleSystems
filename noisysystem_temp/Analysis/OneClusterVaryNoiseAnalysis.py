import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from particle.plotting import anim_torus
from particle.processing import get_master_yaml, match_parameters, load_traj_data
from particle.statistics import calculate_l1_convergence

# Standard plotting choices
# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    "gamma": 0.01,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 2000.0,
    # "dt": 0.015,
}


final_plot_time = 5000000

yaml_path = "../Experiments/one_cluster_vary_noise_scale_dt"
data_path = "../Experiments/Data.nosync/"

history = get_master_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)


fig, ax = plt.subplots()

# Create colour bar and scale
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0, vcenter=0.25, vmax=0.5)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
# Set tick locations
cbar = fig.colorbar(scalarMap, ticks=np.arange(0, 0.5, 0.05))


# For each matching desired parameters, calculate the l1 error and plot
for file_name in file_names:
    simulation_parameters = history[file_name]
    t, error = calculate_l1_convergence(
        file_name,
        plot_hist=False,
        yaml_path=yaml_path,
        data_path=data_path,
        final_plot_time=final_plot_time,
    )

    ax.semilogx(
        t,
        error,
        color=scalarMap.to_rgba(simulation_parameters["D"]),
        label=f"{simulation_parameters['D']}",
        alpha=0.5,
    )
    if simulation_parameters["D"] == 0.05:
        _t, _x, _v = load_traj_data(file_name, simulation_parameters, data_path)

ax.plot([0, t[-1]], [7.5, 7.5], "k--", alpha=0.2)
ax.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
cbar.set_label(r"Diffusion $\sigma$", rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.tight_layout()
# plt.show()
# ani = anim_torus(_t, _x, _v, subsample=50)
plt.show()

# fig.savefig(f"OneClusterVaryNoiselargerGamma.jpg", dpi=300)
