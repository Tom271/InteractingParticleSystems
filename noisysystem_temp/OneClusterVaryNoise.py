import numpy as np
import particle.processing as processing


particles = 1000
diffusion = np.arange(0.05, 0.55, 0.05)

test_params = {
    "particle_count": 10 * [particles],  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": [0.05],
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": diffusion.tolist(),
    "phi": ["Gamma"],
    "initial_dist_x": ["one_cluster"],
    "initial_dist_v": ["pos_normal_dn"],
    "T_end": [200.0],
    "option": ["numba"],
    # "dt": [0.015],
}


history = processing.get_master_yaml(yaml_path="experiments_ran")
fn = """one_cluster_vary_noise_high_particles"""
processing.run_experiment(test_params, history, experiment_name=fn)
