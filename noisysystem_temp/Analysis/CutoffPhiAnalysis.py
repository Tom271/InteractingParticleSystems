from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_helper import plot_convergence_from_clusters, plot_avg_vel

# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    "gamma": 0.1,
    # "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 2000.0,
    "dt": 0.01,
    "D": 0.01,
}

yaml_path = "../Experiments/cutoff_phi_no_of_clusters"

fig, [ax1, ax2] = plt.subplots(1, 2, sharex=True)
ax1 = plot_convergence_from_clusters(ax1, search_parameters, yaml_path)
ax1.plot([0, search_parameters["T_end"]], [7.5, 7.5], "k--", alpha=0.2)

ax2 = plot_avg_vel(ax2, search_parameters, exp_yaml=yaml_path)
ax2.set(ylabel=r"$M^N(t) $")

fig.savefig(f"CutOffPhiConvergence.jpg", dpi=300)
plt.show()
