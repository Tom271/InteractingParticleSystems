from matplotlib import rc
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from analysis_helper import calculate_l1_convergence, plot_avg_vel
from particle.processing import get_master_yaml, match_parameters

# Standard plotting choices
# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    # "gamma": 0.13,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_normal_dn",
    # "T_end": 250.0,
    "dt": 0.005,
    # "D": 0.01,
}

# Path to YAML file relative to current directory
yaml_path = "../Experiments/one_cluster_vary_gamma_higher_noise"
history = get_master_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)


fig, [ax1, ax2] = plt.subplots(1, 2, sharex=True)
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0.06, vcenter=0.125, vmax=0.21)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)

for file_name in file_names:
    print(file_name)
    t, error = calculate_l1_convergence(file_name, plot_hist=False)
    ax1.semilogx(
        t,
        error,
        color=scalarMap.to_rgba(history[file_name]["gamma"]),
        label=f"{history[file_name]['gamma']}",
        alpha=0.5,
    )

ax2 = plot_avg_vel(ax2, search_parameters, exp_yaml=yaml_path, scalarMap=scalarMap)
ax2.set(ylabel=r"$M^N(t) $")

ax1.plot([0, t[-1]], [7.5, 7.5], "k--", alpha=0.2)
ax1.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
cbar = fig.colorbar(scalarMap, ticks=np.arange(0, 0.25, 0.025))
cbar.set_label(r"Interaction $\gamma$", rotation=270)
cbar.ax.get_yaxis().labelpad = 15
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryGamma.jpg", dpi=300)
