# from matplotlib import rc
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

from particle.statistics import (
    calculate_avg_vel,
    calculate_l1_convergence,
    moving_average,
)
from particle.processing import (
    get_master_yaml,
    get_parameter_range,
    match_parameters,
    load_traj_data,
)

# Standard plotting choices
# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "scaling": "Local",
    "D": 0.25,
    "phi": "Gamma",
    "dt": 0.005,
    "G": "Smooth",
    "option": "numba",
    "initial_dist_x": "one_cluster",
    "T_end": 200.0,
    "initial_dist_v": "pos_normal_dn",
    "particle_count": 600,
}  # {"particle_count": 600}
# os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
# os.chdir("E:/")
os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

# Path to YAML file relative to current directory
yaml_path = "./Experiments/one_cluster_vary_gamma_50_runs_higher_particles"
# "../Experiments/one_cluster_low_gamma_ten_runs"
history = get_master_yaml(yaml_path)

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 3), sharex=True)
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0.01, vcenter=0.05, vmax=0.25)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
gammas = get_parameter_range("gamma", history)
# np.array([0.25])  # np.arange(0.05, 0.15, 0.05)

# np.concatenate(([0.01], np.arange(0.05, 0.3, 0.05)))

# np.arange(0.01, 0.3, 0.05)

#  np.concatenate(
#     ([0.01], np.arange(0.05, 0.2, 0.05))
# )
for gamma in gammas:
    search_parameters["gamma"] = gamma
    file_names = match_parameters(search_parameters, history)
    for idx, file_name in enumerate(file_names):
        print(file_name)
        t, x, v = load_traj_data(file_name, data_path="Experiments/Data.nosync/")
        error = calculate_l1_convergence(t, x, v)
        avg_vel = calculate_avg_vel(t, x, v)
        if idx == 0:
            avg_vel_store = np.zeros((len(file_names), len(avg_vel)))
            error_store = np.zeros((len(file_names), len(error)))
        ax1.plot(
            t,
            error,
            color=scalarMap.to_rgba(history[file_name]["gamma"]),
            label=f"{history[file_name]['gamma']}",
            alpha=0.1,
            zorder=1,
        )
        ax2.plot(
            t,
            avg_vel,
            color=scalarMap.to_rgba(history[file_name]["gamma"]),
            label=f"{history[file_name]['gamma']}",
            alpha=0.1,
            zorder=1,
        )
        error_store[idx, :] = error
        avg_vel_store[idx, :] = avg_vel

    # ax1.plot(
    #     t,
    #     np.mean(error_store, axis=0),
    #     color=scalarMap.to_rgba(history[file_name]["gamma"]),
    #     label=f"{history[file_name]['gamma']}",
    #     alpha=0.8,
    #     zorder=2,
    # )
    #
    # ax2.plot(
    #     t,
    #     np.mean(avg_vel_store, axis=0),
    #     color=scalarMap.to_rgba(history[file_name]["gamma"]),
    #     label=f"{history[file_name]['gamma']}",
    #     alpha=0.8,
    #     zorder=2,
    # )
expected_errors = {
    "480": 7.52,
    "600": 6.69,
    "700": 6.26,
    "1000": 5.25,
}
exp_error = expected_errors[str(search_parameters["particle_count"])]
ax1.plot([0, t[-1]], [exp_error, exp_error], "k--", alpha=0.2)
ax1.plot(
    t[19:], moving_average(np.mean(error_store, axis=0), n=20), "r",
)
ax2.plot([0, t[-1]], [1, 1], "k--", alpha=0.2)
ax2.plot(
    t[19:], moving_average(np.mean(avg_vel_store, axis=0), n=20), "r",
)
print(
    f"Final difference in distance is {moving_average(np.mean(error_store, axis=0), n=20)[-1] - exp_error}"
)
print(
    f"Final difference in velocity is {1- moving_average(np.mean(avg_vel_store, axis=0), n=20)[-1]}"
)

ax1.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
ax2.set(xlabel="Time", ylabel=r"$\bar{M}^N(t)$")
# cbar = fig.colorbar(scalarMap, ticks=np.arange(0, max(gammas), 0.05))
# cbar.set_label(r"Interaction $\gamma$", rotation=270)
# cbar.ax.get_yaxis().labelpad = 15
plt.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.9, wspace=0.23)
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryGamma_longrun_log.jpg", dpi=300)
