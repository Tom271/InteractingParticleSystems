from matplotlib import rc
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis_helper import calculate_l1_convergence
from particle.processing import get_master_yaml, match_parameters

# Standard plotting choices
rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    "gamma": 0.01,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 250.0,
    "dt": 0.005,
}


final_plot_time = 50

yaml_path = "../Experiments/one_cluster_vary_noise"
data_path = "../Experiments/Data/"

history = get_master_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)


t = np.arange(
    0, min(search_parameters["T_end"], final_plot_time), search_parameters["dt"]
)

fig, ax = plt.subplots()

# Create colour bar and scale
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0, vcenter=0.25, vmax=0.5)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
# Set tick locations
cbar = fig.colorbar(scalarMap, ticks=np.arange(0, 0.5, 0.05))


# For each matching desired parameters, calculate the l1 error and plot
for file_name in file_names:
    error = calculate_l1_convergence(
        search_parameters,
        file_name,
        plot_hist=False,
        yaml_path=yaml_path,
        data_path=data_path,
        final_plot_time=final_plot_time,
    )
    ax.plot(
        t,
        error,
        color=scalarMap.to_rgba(history[file_name]["D"]),
        label=f"{history[file_name]['D']}",
        alpha=0.5,
    )

ax.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
cbar.set_label(r"Diffusion $\sigma$", rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryNoiselargerGamma.jpg", dpi=300)
