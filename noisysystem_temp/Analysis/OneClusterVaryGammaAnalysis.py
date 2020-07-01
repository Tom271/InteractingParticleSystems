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
    # "gamma": 0.1,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 250.0,
    "dt": 0.005,
    "D": 0.01,
}

# Path to YAML file relative to current directory
yaml_path = "../Experiments/one_cluster_vary_gamma"
history = get_master_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)

t = np.arange(0, search_parameters["T_end"], search_parameters["dt"])

fig, ax = plt.subplots()
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0, vcenter=0.05, vmax=0.1)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

for file_name in file_names:
    print(file_name)
    error = calculate_l1_convergence(search_parameters, file_name, plot_hist=False)
    ax.plot(
        t,
        error,
        color=scalarMap.to_rgba(history[file_name]["gamma"]),
        label=f"{history[file_name]['gamma']}",
        alpha=0.5,
    )

ax.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
cbar = fig.colorbar(scalarMap, ticks=np.arange(0, 0.1, 0.01))
cbar.set_label(r"Interaction $\gamma$", rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryGamma.jpg", dpi=300)
