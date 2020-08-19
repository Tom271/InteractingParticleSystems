import numpy as np
import os
import particle.processing as processing


particles = [168]
# gammas = np.concatenate(([0.01], np.arange(0.05, 0.2, 0.05)))
gammas = np.array([0.05])
#  np.array(
#     [0.01, 0.05, 0.1, 0.15, 0.2]
# )

test_params = {
    "particle_count": 10 * particles,  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": gammas.tolist(),
    "G": ["Smooth", "Step"],
    "scaling": ["Local"],
    "D": [0.0],
    "phi": ["Gamma"],
    "initial_dist_x": ["two_clusters_2N_N"],
    "initial_dist_v": ["2N_N_cluster_const"],
    "T_end": [100.0],
    "dt": [0.005],
    "option": ["numba"],
}


history = processing.get_master_yaml(yaml_path="experiments_ran")
# fn = (
#     f"""{test_params["initial_dist_v"][0]}_"""
#     f"""vel_{test_params["scaling"][0]}_G{test_params["G"][0]}_"""
#     f"""T{int(test_params["T_end"][0])}_noise_report_Galpha"""
# )
os.chdir("E:/")

fn = "2NN_cluster_low_gamma_N168"
processing.run_experiment(test_params, history, experiment_name=fn)
print(
    "Ran for RHS Fig 2", test_params,
)
