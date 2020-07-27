import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import seaborn as sns

import matplotlib.cm as mplcm
import matplotlib.colors as colors

from particle.statistics import calculate_l1_convergence
from particle.processing import get_master_yaml ,match_parameters, load_traj_data

fig = plt.figure(figsize=(12, 4))
# fig.patch.set_alpha(0.0)

grid = plt.GridSpec(4, 3 , wspace=0.33)  # , hspace=0.5)
short_time_ax = plt.subplot(grid[:,0])
long_time_ax_1 = plt.subplot(grid[0,1:])
long_time_ax_2 = plt.subplot(grid[1,1:],sharex=long_time_ax_1,sharey=long_time_ax_1)
long_time_ax_3 = plt.subplot(grid[2,1:],sharex=long_time_ax_1,sharey=long_time_ax_1)
long_time_ax_4 = plt.subplot(grid[3,1:],sharex=long_time_ax_1,sharey=long_time_ax_1)
long_time_axes = [long_time_ax_1,long_time_ax_2,long_time_ax_3,long_time_ax_4]

search_parameters = {
    # "particle_count": 480,
    # "G": "Smooth",
    # "scaling": "Local",
    # "phi": "Bump",
    # # "initial_dist_x": "one_cluster",
    # "initial_dist_v": "pos_const_near_0",
    "T_end": 200.0,
    # "dt": 0.01,
    # "D": 1.0,
    # "option": "numba",
}
if os.name == "nt":
    rc("text", usetex=True)  # I only have TeX on Windows :(
    os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
elif os.name == "posix":
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

# yaml_path = "Experiments/positive_phi_no_of_clusters_high_noise_bump"
yaml_path = "./Experiments/one_cluster_vary_gamma_100_runs"

fn = "_switch_"
logged = False
history = get_master_yaml(yaml_path)
i = 0

gammas = np.arange(0.01, 0.3, 0.05)
break_time_step = 100
cm = plt.get_cmap("coolwarm")
cNorm = colors.BoundaryNorm(gammas, cm.N )
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
for gamma in gammas.tolist():
    search_parameters["gamma"] = gamma

    file_names = match_parameters(search_parameters, history)
    for idx, file_name in enumerate(file_names):
        print(file_name)
        t, error = calculate_l1_convergence(file_name, plot_hist=False)
        t, x, v = load_traj_data(file_name, data_path="Experiments/Data.nosync/")
        avg_vel = v.mean(axis=1)
        if idx == 0:
            avg_vel_store = np.zeros((len(file_names), len(avg_vel)))
            error_store = np.zeros((len(file_names), len(error)))
        short_time_ax.plot(
            t[:break_time_step],
            avg_vel[:break_time_step],
            color=scalarMap.to_rgba(history[file_name]["gamma"]),
            label=f"{history[file_name]['gamma']}",
            alpha=0.1,
            zorder=1,
        )
        error_store[idx, :] = error
        avg_vel_store[idx, :] = avg_vel

        long_time_axes[i].plot(
            t[break_time_step:],
            np.mean(avg_vel_store, axis=0)[break_time_step:],
            color=scalarMap.to_rgba(history[file_name]["gamma"]),
            label=f"{history[file_name]['gamma']}",
            alpha=0.8,
            zorder=2,
        )
    plt.setp(long_time_axes[i].get_xticklabels(), visible=False)
    i += 1
    if i == len(long_time_axes):
        break


short_time_ax.set(xlabel="Time", ylabel=r"Average Velocity $M^N(t)$")
long_time_ax_4.set(xlabel="Time")
sns.set(style="white", context="talk")
handles, labels = short_time_ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
# short_time_ax.legend(handles, labels)
fig.text(0.36, 0.48, r"Average Velocity $M^N(t)$", ha="center", va="center",rotation=90)
plt.show()
