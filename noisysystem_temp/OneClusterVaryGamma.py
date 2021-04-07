import numpy as np
import os
import particle.processing as processing


# particles = [500]
# gammas = np.concatenate(([0.01], np.arange(0.05, 0.2, 0.05)))
# Match PDE Sim
# cluster_width = 0.25
#  np.array(
#     [0.01, 0.05, 0.1, 0.15, 0.2]
# )

# test_params = {
#     "particle_count": 100 * particles,  # (3 * np.arange(8, 150, 16)).tolist(),
#     "gamma": gammas.tolist(),
#     "G": ["Smooth"],
#     "scaling": ["Local"],
#     "D": [0.25],
#     "phi": ["Gamma"],
#     "initial_dist_x": ["wide_PDE_cluster"],
#     "initial_dist_v": ["PDE_normal"],
#     "T_end": [200.0],
#     "dt": [0.005],
#     "option": ["numba"],
#     "record_time": [0.25],
# }

# cluster_width = 0.1
# gammas = np.array([1 / 6, 1 / 3, 1 / 2, 2 / 3, 1, 4 / 3, 5]) * cluster_width

sim_parameters = {
    "particle_count": (3 * np.arange(8, 150, 16)).tolist(),
    "G": ["Alpha Smooth"],
    "scaling": ["Local"],
    "D": [0.0],
    "gamma": np.arange(0.05, 0.55, 0.05).tolist(),
    "phi": ["Gamma"],
    "initial_dist_x": ["det_two_clusters_2N_N"],
    "initial_dist_v": ["2N_N_cluster_const"],
    "T_end": [200],
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
# os.chdir("D:/InteractingParticleSystems/det_system")
os.chdir("/exports/eddie/scratch/s1415551")
fn = "2NN_N_cluster_SteepSmoothG"
processing.run_experiment(sim_parameters, experiment_name=fn)
print(
    "Ran for steep smooth G", sim_parameters,
)
