import particle.processing as processing


particles = 100

test_params = {
    "particle_count": 100 * [particles],  # (3 * np.arange(8, 150, 16)).tolist(),
    # "gamma": (np.arange(0.2, 0.5, 0.05)).tolist(),
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": [1.0],
    "phi": ["Gamma"],
    "gamma": [0.1],
    "initial_dist_x": ["uniform_dn"],
    "initial_dist_v": ["pos_normal_dn"],
    "T_end": [2000.0],
    "dt": [0.01],
    "option": ["numba"],
}


history = processing.get_master_yaml(yaml_path="experiments_ran")
# fn = (
#     f"""{test_params["initial_dist_v"][0]}_"""
#     f"""vel_{test_params["scaling"][0]}_G{test_params["G"][0]}_"""
#     f"""T{int(test_params["T_end"][0])}_noise_report_Galpha"""
# )
fn = f"""PSMean0"""
processing.run_experiment(test_params, history, experiment_name=fn)
