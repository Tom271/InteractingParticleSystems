""" SLOW!!!"""
import numpy as np
import os
import particle.processing as processing


@np.testing.dec.slow
def run_timestep_experiment_high_gamma():
    timesteps = np.logspace(start=0, stop=-4, num=15)
    test_params = {
        "particle_count": 10 * [99, 501],  # (3 * np.arange(8, 150, 16)).tolist(),
        "gamma": [0.5],
        "G": ["Smooth"],
        "scaling": ["Local"],
        "D": [0.25],
        "phi": ["Gamma"],
        "initial_dist_x": ["two_clusters_2N_N"],
        "initial_dist_v": ["2N_N_cluster_const"],
        "T_end": [200.0],
        "dt": timesteps.tolist(),
        "option": ["numba"],
    }

    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path="timestep_experiments")
    fn = "HighGammaLoweringTimestep"
    processing.run_experiment(test_params, history, experiment_name=fn)
    print(
        "Running reduced timestep with gamma =0.5 --- post-process to check against phi=1"
    )


@np.testing.dec.slow
def run_timestep_experiment_phi_uniform():
    timesteps = np.logspace(start=0, stop=-4, num=15)
    test_params = {
        "particle_count": 10 * [99, 501],  # (3 * np.arange(8, 150, 16)).tolist(),
        "G": ["Smooth"],
        "scaling": ["Local"],
        "D": [0.25],
        "phi": ["Uniform"],
        "initial_dist_x": ["two_clusters_2N_N"],
        "initial_dist_v": ["2N_N_cluster_const"],
        "T_end": [200.0],
        "dt": timesteps.tolist(),
        "option": ["numba"],
    }

    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path="timestep_experiments")
    fn = "UniformInteractionLoweringTimestep"
    processing.run_experiment(test_params, history, experiment_name=fn)
    print(
        "Running reduced timestep with phi=1 --- post-process to check against gamma =0.5"
    )


@np.testing.dec.slow
def run_timestep_experiment_low_gamma_low_particles():
    timesteps = np.logspace(start=0, stop=-4, num=15)
    test_params = {
        "particle_count": 20 * [99],  # (3 * np.arange(8, 150, 16)).tolist(),
        "gamma": [0.01],
        "G": ["Smooth"],
        "scaling": ["Local"],
        "D": [0.25],
        "phi": ["Gamma"],
        "initial_dist_x": ["two_clusters_2N_N"],
        "initial_dist_v": ["2N_N_cluster_const"],
        "T_end": [200.0],
        "dt": timesteps.tolist(),
        "option": ["numba"],
    }

    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path="timestep_experiments")
    fn = "LowGammaLoweringTimestepLowParticles"
    processing.run_experiment(test_params, history, experiment_name=fn)
    print(
        "Running reduced timestep with gamma =0.01 and N=100 --- does non-uniformity persist?"
    )


def run_timestep_experiment_low_gamma_high_particles():
    timesteps = np.logspace(start=0, stop=-4, num=15)
    test_params = {
        "particle_count": 20 * [501],
        "gamma": [0.01],
        "G": ["Smooth"],
        "scaling": ["Local"],
        "D": [0.25],
        "phi": ["Gamma"],
        "initial_dist_x": ["two_clusters_2N_N"],
        "initial_dist_v": ["2N_N_cluster_const"],
        "T_end": [200.0],
        "dt": timesteps.tolist(),
        "option": ["numba"],
    }

    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path="timestep_experiments")
    fn = "LowGammaLoweringTimestepHighParticles"
    processing.run_experiment(test_params, history, experiment_name=fn)
    print(
        "Running reduced timestep with gamma =0.01 and N=500 --- does non-uniformity persist?"
    )


if __name__ == "__main__":
    run_timestep_experiment_low_gamma_high_particles()
