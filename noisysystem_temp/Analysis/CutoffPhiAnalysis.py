from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from analysis_helper import plot_convergence_from_clusters

rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    "gamma": 0.1,
    # "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 500.0,
    "dt": 0.01,
    "D": 0.01,
}

yaml_path = "../Experiments/cutoff_phi_no_of_clusters"

fig, ax = plt.subplots()
ax = plot_convergence_from_clusters(ax, search_parameters, yaml_path)
fig.savefig(f"CutOffPhiConvergence.jpg", dpi=300)
plt.show()
