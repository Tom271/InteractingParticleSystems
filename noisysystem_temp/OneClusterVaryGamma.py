import numpy as np
import os
import particle.processing as processing


particles = [500]
# gammas = np.concatenate(([0.01], np.arange(0.05, 0.2, 0.05)))
gammas = np.array([1 / 6, 1 / 3, 1 / 2, 2 / 3, 1, 4 / 3, 2]) * 0.25
#  np.array(
#     [0.01, 0.05, 0.1, 0.15, 0.2]
# )

test_params = {
    "particle_count": 100 * particles,  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": gammas.tolist(),
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": [0.25],
    "phi": ["Gamma"],
    "initial_dist_x": ["wide_PDE_cluster"],
    "initial_dist_v": ["PDE_normal"],
    "T_end": [200.0],
    "dt": [0.005],
    "option": ["numba"],
    "record_time": [0.25],
}


# history = processing.get_master_yaml(yaml_path="experiments_ran")
# fn = (
#     f"""{test_params["initial_dist_v"][0]}_"""
#     f"""vel_{test_params["scaling"][0]}_G{test_params["G"][0]}_"""
#     f"""T{int(test_params["T_end"][0])}_noise_report_Galpha"""
# )
# os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
os.chdir("/exports/eddie/scratch/s1415551")
fn = "test_one_cluster_match_PDE"
processing.run_experiment(test_params, experiment_name=fn)
print(
    "Ran for PDE Setup", test_params,
)
