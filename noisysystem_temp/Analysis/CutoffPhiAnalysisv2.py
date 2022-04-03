import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import os
import seaborn as sns
from particle.plotting import _get_number_of_clusters
from particle.processing import match_parameters, get_main_yaml, load_traj_data
from particle.statistics import (
    calculate_avg_vel,
    # calculate_l1_convergence,
    corrected_calculate_l1_convergence,
)

sns.color_palette("colorblind")
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 480,
    "G": "Smooth",
    "scaling": "Local",
    "phi": "Normalised Gamma",
    "gamma": 0.05,
    # "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_const_near_0",
    "T_end": 2000.0,
    "dt": 0.01,
    "D": 1.0,
}
if os.name == "nt":
    rc("text", usetex=True)  # I only have TeX on Windows :(
    os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
elif os.name == "posix":
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

yaml_path = "Experiments/cutoff_phi_normalised_gamma"  # cutoff_phi_no_of_clusters_ten_runs_higher_noise_smaller_gamma_long_run"
data_path = "Experiments/Data.nosync/"  # parquet_data/"
history = get_main_yaml(yaml_path)
vel_break_time_step = 40
l1_break_time_step = 20
parameter_range = ["one_cluster", "two_clusters", "three_clusters", "four_clusters"]
include_traj = True

fig = plt.figure(figsize=(20, 6))
grid = plt.GridSpec(
    len(parameter_range), 4, wspace=0.35, bottom=0.2, left=0.05, right=0.95
)

l1_ax = fig.add_subplot(grid[:, 0])
short_time_ax = fig.add_subplot(grid[:, 1])
long_time_axes = []
for idx, elem in enumerate(parameter_range):
    try:
        long_time_axes.append(
            fig.add_subplot(
                grid[idx, 2:], sharey=long_time_axes[0], sharex=long_time_axes[0]
            )
        )
    except IndexError:
        long_time_axes.append(fig.add_subplot(grid[idx, 2:]))
    if idx != len(parameter_range) - 1:
        plt.setp(long_time_axes[idx].get_xticklabels(), visible=False)

# Reverse so that plots line up with colorbar
long_time_axes = long_time_axes[::-1]

# Create colorbar and labels
fig.text(
    0.49,
    0.53,
    r"Average Velocity $\bar{M}^N(t)$",
    ha="center",
    va="center",
    rotation=90,
)
short_time_ax.set(xlabel="Time", ylabel=r"Average Velocity $M^N(t)$")
l1_ax.set(xlabel="Time", ylabel=r"$\bar{\ell}^1$ Distance")
long_time_axes[0].set(xlabel="Time")

cluster_colour = ["#0571b0", "#92c5de", "#f4a582", "#ca0020"]

# Populate the plots
for idx, parameter_value in enumerate(parameter_range):
    search_parameters["initial_dist_x"] = parameter_value
    file_names = match_parameters(search_parameters, history, exclude={"dt": [1.0]})
    if not file_names:
        print("Skipping...")
        continue
    metric_store = []
    l1_store = []
    if len(file_names) > 15:
        file_names = file_names[-15:]

    for file_name in file_names:
        simulation_parameters = history[file_name]
        cluster_count = _get_number_of_clusters(simulation_parameters["initial_dist_x"])
        colour = cluster_colour[cluster_count - 1]
        cluster_label = f"{cluster_count} cluster{'' if cluster_count==1 else 's'}"

        t, x, v = load_traj_data(file_name, data_path=data_path)
        metric_result = calculate_avg_vel(t, x, v)
        l1_result = corrected_calculate_l1_convergence(t, x, v)
        metric_store.append(metric_result)
        l1_store.append(l1_result)

        short_time_ax.plot(
            t[:vel_break_time_step],
            metric_result[:vel_break_time_step],
            label=cluster_label,
            color=colour,
            alpha=0.1,
            zorder=1,
        )

        if include_traj:
            long_time_axes[idx].plot(
                t[vel_break_time_step:],
                metric_result[vel_break_time_step:],
                color=colour,
                label=cluster_label,
                alpha=0.05,
                zorder=1,
            )

    metric_store = np.array(metric_store)
    l1_store = np.array(l1_store)
    l1_ax.plot(
        t[:l1_break_time_step],
        l1_store.mean(axis=0)[:l1_break_time_step],
        color=colour,
        label=cluster_label,
        alpha=1,
        zorder=2,
    )

    long_time_axes[idx].plot(
        t[vel_break_time_step:],
        metric_store.mean(axis=0)[vel_break_time_step:],
        color=colour,
        label=cluster_label,
        alpha=1,
        zorder=2,
    )

    long_time_axes[idx].plot(
        [t[vel_break_time_step], t[-1]],
        [np.sign(metric_result[0]), np.sign(metric_result[0])],
        "k--",
        alpha=0.25,
    )

handles, labels = l1_ax.get_legend_handles_labels()
# sort both labels and handles by labels
labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
l1_ax.legend(handles, labels)
# expected_error = {
#     "480": 7.45637,
#     "500": 7.39326,
#     "600": 6.64672,
#     "700": 6.26729,
#     "1000": 5.26257,
# }
expected_error = {
    "480": 0.2770097916666669,
    "500": 0.2752154,
    "600": 0.2481263333333336,
    "700": 0.23227776190476193,
    "1000": 0.1942373,
}
l1_ax.plot(
    [0, t[l1_break_time_step]],
    [
        expected_error[str(simulation_parameters["particle_count"])],
        expected_error[str(simulation_parameters["particle_count"])],
    ],
    "k--",
    alpha=0.25,
)

fig.savefig(f"CutoffPhiAnalysisL1AvgVelMultiTimescale.jpg", dpi=300)
plt.show()
