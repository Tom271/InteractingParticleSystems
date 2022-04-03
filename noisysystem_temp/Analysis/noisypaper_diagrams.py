# from typing import Match
# import matplotlib.cm as mplcm
# import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import rc

# import numpy as np
import os

from particle.plotting import multiple_timescale_plot
from particle.processing import get_main_yaml, get_parameter_range
from particle.statistics import (
    calculate_avg_vel,
    # calculate_l1_convergence,
    corrected_calculate_l1_convergence,
)


def HigherParticlesFig():
    search_parameters = {
        "initial_dist_v": "pos_normal_dn",
        "initial_dist_x": "one_cluster",
        "dt": 0.005,
        "option": "numba",
        "G": "Smooth",
        "phi": "Gamma",
        "D": 0.25,
        "scaling": "Local",
        "particle_count": 700,
        "T_end": 200.0,
    }
    os.chdir("D:/2907Data")
    # Path to YAML file relative to current directory
    yaml_path = "./Experiments/one_cluster_vary_gamma_50_runs_higher_particles"
    history = get_main_yaml(yaml_path)
    gammas = get_parameter_range("gamma", history)
    fig = multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=corrected_calculate_l1_convergence,
        parameter="gamma",
        parameter_range=gammas,
        history=history,
        include_traj=False,
    )
    fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
    fig.savefig(
        f"OneClusterVaryGammaMulti{search_parameters['particle_count']}.jpg", dpi=300
    )
    plt.show()
    return


def OneClusterVaryGammaFig():
    search_parameters = {
        "initial_dist_v": "pos_normal_dn",
        "initial_dist_x": "one_cluster",
        "dt": 0.005,
        "option": "numba",
        "G": "Smooth",
        "phi": "Normalised Gamma",
        "D": 0.25,  # 0.5,
        "scaling": "Global",  # "Global",
        "particle_count": 480,
        "T_end": 200.0,
    }
    if os.name == "nt":
        rc("text", usetex=True)  # I only have TeX on Windows :(
        os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
    elif os.name == "posix":
        os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

    # os.chdir("D:/2907Data")yaml_path = "./Experiments/one_cluster_vary_gamma_neg_mean"
    # os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

    # Path to YAML file relative to current directory
    yaml_path = "./Experiments/OneClusterVaryGammaGlobal"
    history = get_main_yaml(yaml_path)
    gammas = get_parameter_range("gamma", history)

    metric_fn = corrected_calculate_l1_convergence
    fig = multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=metric_fn,
        parameter="gamma",
        parameter_range=gammas,
        history=history,
        include_traj=False,
        plot_short_average=True,
        data_path="Experiments/Data.nosync/",
    )
    # fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
    fig.savefig(
        f"OneClusterVarySmallGammaMulti{search_parameters['initial_dist_v']}{metric_fn.__name__[9:]}average_lower_noise_v3.jpg",
        dpi=300,
    )
    plt.show()
    return


def OneClusterVaryNoiseFig():
    search_parameters = {
        "initial_dist_v": "pos_normal_dn",
        "initial_dist_x": "one_cluster",
        # "dt": 0.005,
        "option": "numba",
        "G": "Smooth",
        "phi": "Normalised Gamma",
        "scaling": "Local",
        "particle_count": 480,
        "T_end": 200.0,
    }
    if os.name == "nt":
        rc("text", usetex=True)  # I only have TeX on Windows :(
        os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
    elif os.name == "posix":
        os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

    # os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")
    # Path to YAML file relative to current directory
    yaml_path = "./Experiments/OneClusterNormalisedVaryNoiseLocal"
    history = get_main_yaml(yaml_path)
    noises = get_parameter_range("D", history)

    for metric_fn in [calculate_avg_vel]:  # , corrected_calculate_l1_convergence]:
        fig = multiple_timescale_plot(
            search_parameters,
            break_time_step=40,
            metric=metric_fn,
            parameter="D",
            parameter_range=noises,
            history=history,
            include_traj=False,
            data_path="Experiments/Data.nosync/",  # parquet_data
        )
        # fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
        fig.savefig(
            f"OneClusterVaryNoiseMulti{search_parameters['initial_dist_v']}{metric_fn.__name__[9:]}_norm_gamma.jpg",
            dpi=300,
        )
    plt.show()
    return


def MatchPDEOneClusterVaryGammaFig():
    search_parameters = {
        "initial_dist_v": "PDE_normal",
        "initial_dist_x": "wide_PDE_cluster",
        "dt": 0.005,
        "option": "numba",
        "G": "Smooth",
        "phi": "Normalised Gamma",
        "D": 0.25,
        "scaling": "Local",
        "particle_count": 480,
        "T_end": 200.0,
    }
    if os.name == "nt":
        rc("text", usetex=True)  # I only have TeX on Windows :(
        os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
    elif os.name == "posix":
        os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

    # Path to YAML file relative to current directory
    yaml_path = "Experiments/OneClusterVaryNormalisedGammaLocal"
    history = get_main_yaml(yaml_path)
    gammas = get_parameter_range("gamma", history)

    for metric_fn in [calculate_avg_vel, corrected_calculate_l1_convergence]:
        fig = multiple_timescale_plot(
            search_parameters,
            break_time_step=80,
            metric=metric_fn,
            parameter="gamma",
            parameter_range=gammas,
            history=history,
            include_traj=False,
            plot_short_average=True,
            data_path="Experiments/Data.nosync/",
        )
        # fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
        fig.savefig(
            f"OneClusterVaryNormalisedGammaMulti{search_parameters['initial_dist_v']}{metric_fn.__name__[9:]}highlightaverage_and_traj_lower_noise.jpg",
            dpi=300,
        )
    plt.show()
    # plt.subplots_adjust(left=0.07, right=0.97, bottom=0.15, top=0.9, wspace=0.23)
    # plt.tight_layout()
    # plt.show()
    return


if __name__ == "__main__":
    # OneClusterVaryGammaFig()
    # MatchPDEOneClusterVaryGammaFig()
    OneClusterVaryNoiseFig()  # tested with normalised_gamma -- none available
