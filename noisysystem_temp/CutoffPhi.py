import particle.processing_v2 as processing


particles = 480


test_params = {
    "particle_count": [particles],  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": [0.1],
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": [0.01],
    "phi": ["Gamma"],
    "initial_dist_x": [
        "one_cluster",
        "two_clusters",
        "three_clusters",
        "four_clusters",
    ],
    "initial_dist_v": ["pos_const_near_0"],
    "T_end": [500.0],
    "dt": [0.01],
    "option": ["numba"],
}


history = processing.get_master_yaml(filename="experiments_ran")
# fn = (
#     f"""{test_params["initial_dist_v"][0]}_"""
#     f"""vel_{test_params["scaling"][0]}_G{test_params["G"][0]}_"""
#     f"""T{int(test_params["T_end"][0])}_noise_report_Galpha"""
# )
fn = f"""cutoff_phi_no_of_clusters"""
processing.run_experiment(test_params, history, experiment_name=fn)