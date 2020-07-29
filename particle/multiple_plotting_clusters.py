import matplotlib.pyplot as plt
from matplotlib import rc
import os
import seaborn as sns
from particle.plotting import plot_avg_vel, plot_averaged_avg_vel
from particle.processing import get_master_yaml, match_parameters

fig = plt.figure(figsize=(12, 4))
# fig.patch.set_alpha(0.0)

grid = plt.GridSpec(4, 3, wspace=0.33)
short_time_ax = plt.subplot(grid[:, 0])
long_time_ax_1 = plt.subplot(grid[0, 1:])
long_time_ax_2 = plt.subplot(grid[1, 1:], sharex=long_time_ax_1, sharey=long_time_ax_1)
long_time_ax_3 = plt.subplot(grid[2, 1:], sharex=long_time_ax_1, sharey=long_time_ax_1)
long_time_ax_4 = plt.subplot(grid[3, 1:], sharex=long_time_ax_1, sharey=long_time_ax_1)
long_time_axes = [long_time_ax_1, long_time_ax_2, long_time_ax_3, long_time_ax_4]

search_parameters = {
    # "particle_count": 480,
    # "G": "Smooth",
    # "scaling": "Local",
    "phi": "Bump",
    # # "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 2000.0,
    # "dt": 0.01,
    # "D": 1.0,
    # "option": "numba",
}
if os.name == "nt":
    rc("text", usetex=True)  # I only have TeX on Windows :(
    os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
elif os.name == "posix":
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

yaml_path = "Experiments/positive_phi_no_of_clusters_high_noise_bump"
fn = "_switch_"
logged = False
history = get_master_yaml(yaml_path)
i = 0


for initial_dist_x in [
    "one_cluster",
    "two_clusters",
    "three_clusters",
    "four_clusters",
]:
    search_parameters["initial_dist_x"] = initial_dist_x
    file_names = match_parameters(search_parameters, history)
    include_traj = True
    # long_time_axes[i] = plot_avg_vel(long_time_axes[i], search_parameters, logx=logged, exp_yaml=yaml_path, end_time_step=100)
    long_time_axes[i] = plot_averaged_avg_vel(
        long_time_axes[i],
        search_parameters,
        logx=logged,
        exp_yaml=yaml_path,
        include_traj=True,
        start_time_step=200,
        end_time_step=1000,
    )
    short_time_ax = plot_averaged_avg_vel(
        short_time_ax,
        search_parameters,
        logx=False,
        exp_yaml=yaml_path,
        include_traj=True,
        end_time_step=30,
    )
    i += 1


plt.setp(long_time_ax_1.get_xticklabels(), visible=False)
plt.setp(long_time_ax_2.get_xticklabels(), visible=False)
plt.setp(long_time_ax_3.get_xticklabels(), visible=False)

short_time_ax.set(xlabel="Time", ylabel=r"Average Velocity $M^N(t)$")
long_time_ax_4.set(xlabel="Time")
sns.set(style="white", context="talk")
handles, labels = short_time_ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
short_time_ax.legend(handles, labels)
fig.text(
    0.36, 0.48, r"Average Velocity $M^N(t)$", ha="center", va="center", rotation=90
)
plt.show()
