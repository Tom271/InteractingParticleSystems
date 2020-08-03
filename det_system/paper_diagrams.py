# from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import matplotlib.cm as mplcm
import matplotlib.colors as colors

from particle.processing import get_master_yaml, match_parameters, load_traj_data

# rc("text", usetex=True)
# sns.set(style="white", context="talk")
"""NEEDS TO BE ON MAC -- FEATHER ENCODING ISSUE """


def phi_one_convergence(time_ax="linear"):
    # sns.set(style="white", context="paper")
    sns.set_style("ticks")
    fig1, ax1 = plt.subplots(figsize=(12, 6))

    cm = plt.get_cmap("coolwarm")
    cNorm = colors.DivergingNorm(vmin=-25, vcenter=0.0, vmax=25)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    yaml_path = "./Experiments/phi_one_convergence"
    history = get_master_yaml(yaml_path)
    file_names = match_parameters({}, history)
    dt = 0.01
    print(yaml_path)

    for file_name in file_names:
        x, v = load_traj_data(file_name, data_path="./Experiments/Data/")
        t = np.arange(0, len(v) * dt, dt)
        if time_ax == "linear":
            ax1.plot(
                t[: int(20 / dt)],
                v[: int(20 / dt)].mean(axis=1),
                color=scalarMap.to_rgba(np.sign(v[0].mean()) * v[0].var()),
            )
        else:
            ax1.semilogx(
                t[: int(20 / dt)],
                v[: int(20 / dt)].mean(axis=1),
                color=scalarMap.to_rgba(np.sign(v[0].mean()) * v[0].var()),
            )
    ax1.plot([0, 20], [1, 1], "k--", alpha=0.25)
    ax1.plot([0, 20], [-1, -1], "k--", alpha=0.25)
    ax1.set(xlabel="Time", ylabel=r"Average Velocity, $M^N(t)$")
    plt.subplots_adjust(
        top=0.905, bottom=0.135, left=0.115, right=0.925, hspace=0.2, wspace=0.2
    )
    plt.show()
    return


def particle_linestyle(particle_count: int):
    if particle_count == 408:
        return "solid"
    elif particle_count == 168:
        return "solid"

    else:
        return "solid"


def avg_vel(
    ax,
    scalarMap,
    file_path: str = "Experiments/Data/",
    yaml_path: str = "Experiments/vary_large_gamma_local",
    search_parameters: dict = {
        "particle_count": 450,
        "G": "Step",
        "scaling": "Local",
        "phi": "Gamma",
        "initial_dist_x": "two_clusters_2N_N",
        "initial_dist_v": "2N_N_cluster_const",
        "T_end": 100,
        "dt": 0.01,
    },
    particle_lines: bool = False,
):

    list_of_names = []
    print(yaml_path)
    history = get_master_yaml(yaml_path)
    list_of_names = match_parameters(search_parameters, history)

    for file_name in list_of_names:
        simulation_parameters = history[file_name]
        _, x, v = load_traj_data(file_name, data_path=file_path)
        t = np.arange(
            0, len(x) * simulation_parameters["dt"], simulation_parameters["dt"]
        )
        if simulation_parameters["gamma"] >= 0.05 and particle_lines:
            ax.semilogx(
                t,
                v.mean(axis=1),
                color=scalarMap.to_rgba(simulation_parameters["gamma"]),
                label="{:.2f}".format(simulation_parameters["gamma"]),
                linestyle=particle_linestyle(simulation_parameters["particle_count"]),
            )
        elif simulation_parameters["gamma"] >= 0.05:
            ax.semilogx(
                t,
                v.mean(axis=1),
                color=scalarMap.to_rgba(simulation_parameters["gamma"]),
                label="{:.2f}".format(simulation_parameters["gamma"]),
            )
    return ax


def plot_gamma_avg_vel():
    sim_parameters = {
        "G": "Step",
        "scaling": "Local",
        "phi": "Gamma",
        "initial_dist_x": "two_clusters_2N_N",
        "initial_dist_v": "2N_N_cluster_const",
        "T_end": 100,
        "dt": 0.005,
    }
    gamma_range = np.arange(0.0, 0.55, 0.05)

    sns.set_style("ticks")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
    plt.subplots_adjust(wspace=0.05)
    cm = plt.get_cmap("coolwarm")
    cNorm = colors.BoundaryNorm(gamma_range, cm.N)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    # cNorm = colors.DivergingNorm(vmin=0, vcenter=0.15, vmax=0.5)
    cbar = fig.colorbar(scalarMap, ax=axes, ticks=gamma_range + 0.025)
    cbar.ax.set_yticklabels([f"{x:.2}" for x in gamma_range])
    cbar.set_label(r"Interaction $\gamma$", rotation=270)
    cbar.ax.get_yaxis().labelpad = 20

    ax1, ax2 = axes
    ax1 = avg_vel(
        ax1,
        scalarMap,
        file_path="Experiments/Data.nosync/",
        yaml_path="Experiments/vary_large_gamma_local",
        search_parameters=sim_parameters,
    )
    sim_parameters["particle_count"] = 408
    ax1 = avg_vel(
        ax1,
        scalarMap,
        file_path="Experiments/Data.nosync/",
        yaml_path="Experiments/vary_small_gamma_local",
        search_parameters=sim_parameters,
    )
    sim_parameters.pop("particle_count")
    sim_parameters["gamma"] = 0.05
    ax2 = avg_vel(
        ax2,
        scalarMap,
        file_path="Experiments/Data.nosync/",
        yaml_path="Experiments/vary_small_gamma_local",
        search_parameters=sim_parameters,
        particle_lines=True,
    )

    ax1.set(xlabel="Time", ylabel=r"Average Velocity $M^N(t)$")
    ax2.set(xlabel="Time")

    for ax in axes:
        ax.plot([0, sim_parameters["T_end"]], [1, 1], "k--", alpha=0.25)
        ax.plot([0, sim_parameters["T_end"]], [-1, -1], "k--", alpha=0.25)
        ax.plot([0, sim_parameters["T_end"]], [0, 0], "k--", alpha=0.25)
    # fig.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    # phi_one_convergence("log")
    if os.name == "nt":
        # rc("text", usetex=True)  # I only have TeX on Windows :(
        os.chdir("D:/InteractingParticleSystems/det_system")
    elif os.name == "posix":
        os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/det_system")

    plot_gamma_avg_vel()
