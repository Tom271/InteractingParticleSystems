import matplotlib.pyplot as plt
import os
from particle.plotting import multiple_timescale_plot
from particle.processing import get_master_yaml, get_parameter_range
from particle.statistics import calculate_avg_vel, calculate_l1_convergence


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
    history = get_master_yaml(yaml_path)
    gammas = get_parameter_range("gamma", history)
    fig = multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=calculate_l1_convergence,
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
        "phi": "Gamma",
        "D": 0.25,
        "scaling": "Local",
        "particle_count": 480,
        "T_end": 200.0,
    }
    # os.chdir("D:/2907Data")yaml_path = "./Experiments/one_cluster_vary_gamma_neg_mean"
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

    # Path to YAML file relative to current directory
    yaml_path = "./Experiments/one_cluster_vary_gamma_100_runs"
    history = get_master_yaml(yaml_path)
    gammas = get_parameter_range("gamma", history)
    gammas_truncated = gammas
    metric_fn = calculate_avg_vel
    fig = multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=metric_fn,
        parameter="gamma",
        parameter_range=gammas_truncated,
        history=history,
        include_traj=False,
    )
    # fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
    fig.savefig(
        f"OneClusterVarySmallGammaMulti{search_parameters['initial_dist_v']}{metric_fn.__name__[9:]}.jpg",
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
        "phi": "Gamma",
        "scaling": "Local",
        "particle_count": 480,
        "T_end": 200.0,
    }
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")
    # Path to YAML file relative to current directory
    yaml_path = "./Experiments/one_cluster_vary_noise_scale_dt_100_runs_larger_gamma"
    history = get_master_yaml(yaml_path)
    noises = get_parameter_range("D", history)

    metric_fn = calculate_avg_vel
    fig = multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=metric_fn,
        parameter="D",
        parameter_range=noises,
        history=history,
        include_traj=False,
    )
    fig.suptitle(f"N = {search_parameters['particle_count']}", size=20)
    fig.savefig(
        f"OneClusterVaryNoiseMulti{search_parameters['initial_dist_v']}{metric_fn.__name__[9:]}.jpg",
        dpi=300,
    )
    plt.show()
    return


if __name__ == "__main__":
    HigherParticlesFig()
