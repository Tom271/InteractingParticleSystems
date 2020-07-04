from matplotlib import cycler
import matplotlib.pyplot as plt
from numba import jit
import numpy as np
from particle.processing import match_parameters, load_traj_data, get_master_yaml
from matplotlib import colors
import matplotlib.cm as mplcm


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def plot_avg_vel(
    ax,
    search_parameters: dict,
    scalarMap=None,
    data_path: str = "../Experiments/Data.nosync/",
    exp_yaml: str = "../Experiments/positive_phi_no_of_clusters",
):
    """Plots average velocity of particles on log scale, colours lines according to
    number of clusters in the inital condition
    """
    if scalarMap is None:
        cm = plt.get_cmap("coolwarm")
        cNorm = colors.DivergingNorm(vmin=0.06, vcenter=0.125, vmax=0.21)
        scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    history = get_master_yaml(exp_yaml)
    list_of_names = match_parameters(search_parameters, history)
    print(list_of_names)
    for file_name in list_of_names:
        simulation_parameters = history[file_name]
        t, x, v = load_traj_data(file_name, data_path)

        cluster_count = _get_number_of_clusters(history[file_name]["initial_dist_x"])
        ax.semilogx(
            t,
            v.mean(axis=1),
            color=scalarMap.to_rgba(history[file_name]["gamma"]),
            # cluster_count),
            # label=f"{cluster_count} Clusters",
            alpha=0.25,
        )
    # plt.tight_layout()
    return ax


def _get_number_of_clusters(initial_condition: str) -> int:
    """Helper to turn initial condition string into int number of clusters"""
    cluster_count = {
        "one_cluster": 1,
        "two_clusters": 2,
        "three_clusters": 3,
        "four_clusters": 4,
    }
    number_of_clusters = cluster_count[initial_condition]
    return number_of_clusters


def calculate_l1_convergence(
    file_name: str,
    plot_hist: bool = False,
    data_path: str = "../Experiments/Data.nosync/",
    yaml_path: str = "../Experiments/",
    final_plot_time: float = 100000,
):
    t, x, v = load_traj_data(file_name, data_path)

    dt = t[1] - t[0]
    error = []
    if plot_hist is True:
        colormap = plt.get_cmap("viridis")
        fig, ax = plt.subplots()
        ax.set_prop_cycle(
            cycler(color=[colormap(k) for k in np.linspace(1, 0, int(1 / dt))])
        )
    for i in np.arange(0, int(min(len(t), final_plot_time // 0.5))):
        hist_x, bin_edges = numba_hist(x, i)
        # hist_x = hist_x / len(x[0, :])
        error_t = np.abs((1 / (2 * np.pi)) - hist_x).sum()
        error.append(error_t)
        if plot_hist is True:
            ax.plot(bin_edges[:-1], hist_x)

    if plot_hist is True:
        ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "k--")
        ax.set(xlim=[0, 2 * np.pi], xlabel="Position", ylabel="Density")
        return t, error, fig, ax
    else:
        return t, error


# @jit(nopython=True, fastmath=True)
def numba_hist(x, i):
    hist_x, bin_edges = np.histogram(
        x[i, :], bins=np.arange(0, 2 * np.pi, np.pi / 60), density=True
    )
    # db = np.pi / 60  # np.diff(bin_edges)
    # hist_x = hist_x / db
    return hist_x, bin_edges


def plot_convergence_from_clusters(ax, search_parameters: dict, yaml_path: str):
    history = get_master_yaml(yaml_path)
    file_names = match_parameters(search_parameters, history)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for file_name in file_names:
        print(file_name)
        simulation_parameters = history[file_name]
        t, error = calculate_l1_convergence(file_name, plot_hist=False)
        cluster_count = _get_number_of_clusters(simulation_parameters["initial_dist_x"])
        ax.plot(
            t, error, label=f"{cluster_count} clusters", color=cycle[cluster_count - 1]
        )

    ax.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    plt.tight_layout()
    return ax
