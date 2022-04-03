import numpy as np
import os
import particle.processing as processing


particles = 480
diffusion = np.arange(0.05, 0.55, 0.05)

test_params = {
    "particle_count": 100 * [particles],  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": [0.042],
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": diffusion.tolist(),
    "phi": ["Normalised Gamma"],
    "initial_dist_x": ["one_cluster"],
    "initial_dist_v": ["pos_normal_dn"],
    "T_end": [200.0],
    "option": ["numba"],
    # "dt": [0.015],
}


# history = processing.get_master_yaml(yaml_path="experiments_ran")
os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
fn = """OneClusterNormalisedVaryNoiseLocal"""
processing.run_experiment(test_params, experiment_name=fn)
print(
    "Ran for Local Scaling", test_params,
)
