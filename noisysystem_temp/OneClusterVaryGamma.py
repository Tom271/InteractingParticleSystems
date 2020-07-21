import numpy as np
import particle.processing as processing


particles = [480]
gammas = np.concatenate(([0.01], np.arange(0.05, 0.2, 0.05)))

test_params = {
    "particle_count": 10 * particles,  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": gammas.tolist(),
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": [0.0],
    "phi": ["Gamma"],
    "initial_dist_x": ["one_cluster"],
    "initial_dist_v": ["pos_const_near_0"],
    "T_end": [200.0],
    "dt": [0.005],
    "option": ["numba"],
}


history = processing.get_master_yaml(yaml_path="experiments_ran")
# fn = (
#     f"""{test_params["initial_dist_v"][0]}_"""
#     f"""vel_{test_params["scaling"][0]}_G{test_params["G"][0]}_"""
#     f"""T{int(test_params["T_end"][0])}_noise_report_Galpha"""
# )
fn = "one_cluster_no_noise_pos_const_near_0"
processing.run_experiment(test_params, history, experiment_name=fn)
# print(
#     "Ran as in Fig 6 but with more particles", test_params,
# )
