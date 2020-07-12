from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns

from particle.plotting import (
    plot_averaged_convergence_from_clusters,
    plot_averaged_avg_vel,
    plot_avg_vel,
)

# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Gamma",
    "gamma": 0.05,
    # "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_normal_dn",
    "T_end": 2000.0,
    "dt": 0.01,
    "D": 1.0,
}

yaml_path = "../Experiments/cutoff_phi_no_of_clusters_ten_runs_higher_noise_smaller_gamma_long_run"

is_logged = True
plot_all = True
fn = "_smaller_gamma_"
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 5), sharex=True)
# ax2 = plot_avg_vel(ax2, search_parameters, logx=is_logged, exp_yaml=yaml_path)
ax2 = plot_averaged_avg_vel(
    ax2, search_parameters, logx=is_logged, include_traj=plot_all, exp_yaml=yaml_path
)

ax1 = plot_averaged_convergence_from_clusters(
    ax1, search_parameters, yaml_path, logx=is_logged
)
ax1.plot([0, search_parameters["T_end"]], [7.5, 7.5], "k--", alpha=0.2)

ax2.set(xlabel="Time", ylabel=r"$M^N(t) $")
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.9, wspace=0.23)
fig.savefig(f"img/CutOffPhiConvergence{fn}logged.jpg", dpi=300)
plt.show()

is_logged = False
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
# ax2 = plot_avg_vel(ax2, search_parameters, logx=is_logged, exp_yaml=yaml_path)
ax2 = plot_averaged_avg_vel(
    ax2, search_parameters, logx=is_logged, include_traj=plot_all, exp_yaml=yaml_path
)

ax1 = plot_averaged_convergence_from_clusters(
    ax1, search_parameters, yaml_path, logx=is_logged
)
ax1.plot([0, search_parameters["T_end"]], [7.5, 7.5], "k--", alpha=0.2)

ax2.set(xlabel="Time", ylabel=r"$M^N(t) $")
plt.tight_layout()
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.9, wspace=0.23)
fig.savefig(f"img/CutOffPhiConvergence{fn}linear.jpg", dpi=300)
plt.show()
