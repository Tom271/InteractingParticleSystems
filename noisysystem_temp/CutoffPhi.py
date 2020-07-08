import particle.processing as processing


particles = 480


test_params = {
    "particle_count": 50 * [particles],  # (3 * np.arange(8, 150, 16)).tolist(),
    "gamma": [0.05],
    "G": ["Smooth"],
    "scaling": ["Local"],
    "D": [1.0],
    "phi": ["Gamma"],
    "initial_dist_x": [
        "one_cluster",
        "two_clusters",
        "three_clusters",
        "four_clusters",
    ],
    "initial_dist_v": ["pos_normal_dn", "pos_const_near_0"],
    "T_end": [2000.0],
    "dt": [0.01],
    "option": ["numba"],
}


history = processing.get_master_yaml(yaml_path="experiments_ran")
fn = "cutoff_phi_no_of_clusters_ten_runs_higher_noise_smaller_gamma_long_run"
processing.run_experiment(test_params, history, experiment_name=fn)
