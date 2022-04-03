from matplotlib import rc

# import matplotlib.animation as animation

import matplotlib.pyplot as plt
import numpy as np
import os

# import seaborn as sns

from particle.plotting import anim_torus
from particle.processing import get_main_yaml, match_parameters, load_traj_data

# Standard plotting choices
rc("text", usetex=True)
# sns.set(style="white", context="talk")

search_parameters = {
    # "particle_count": 480,
    # "G": "Smooth",
    # "scaling": "Local",
    # "phi": "Gamma",
    # "gamma": 0.13,
    # "initial_dist_x": "one_cluster",
    # "initial_dist_v": "pos_normal_dn",
    # # "T_end": 250.0,
    # "dt": 0.005,
    # # "D": 0.01,
}
# search_parameters = {
#     "initial_dist_v": "pos_normal_dn",
#     "initial_dist_x": "one_cluster",
#     # "dt": 0.005,
#     "option": "numba",
#     "G": "Smooth",
#     "D": 0.05,
#     "phi": "Gamma",
#     # "gamma": 0.01,
#     "scaling": "Local",
#     "particle_count": 480,
#     "T_end": 200.0,
# }
# search_parameters = {
#     "initial_dist_v": "pos_normal_dn",
#     "initial_dist_x": "one_cluster",
#     # "dt": 0.005,
#     "option": "numba",
#     "G": "Smooth",
#     "D": 0.25,
#     "phi": "Gamma",
#     "gamma": 0.01,
#     "scaling": "Local",
#     "particle_count": 480,
#     "T_end": 200.0,
# }
#
# search_parameters = {
#     "particle_count": 216,
#     "gamma": 0.05,
#     "G": "Alpha Smooth",
#     "alpha": 100,
#     "scaling": "Local",
#     "phi": "Gamma",
#     "D": 0.0,
#     "initial_dist_x": "two_clusters_2N_N",
#     "initial_dist_v": "2N_N_cluster_const",
#     "T_end": 200,
#     "dt": 0.005,
# }
search_parameters = {
    "particle_count": 2000,
    "D": 0.25,
    "G": "Garnier",
    "alpha": 8,
    "scaling": "Global",
    "phi": "Normalised Gamma",
    "gamma": 0.01,  # 0.1, 0.05, 0.01
    "initial_dist_x": "one_cluster",
    "initial_dist_v": "pos_normal_dn",
    "T_end": 200.0,
    "dt": 0.005,
}
# search_parameters = {
#     "particle_count": 100,  # (3 * np.arange(8, 150, 16)).tolist(),
#     # "gamma": (np.arange(0.2, 0.5, 0.05)).tolist(),
#     "G": "Smooth",
#     "scaling": "Local",
#     "D": 1.0,
#     "phi": "Gamma",
#     "gamma": 0.1,
#     "initial_dist_x": "uniform_dn",
#     "initial_dist_v": "pos_normal_dn",
#     "T_end": 2000.0,
#     "dt": 0.01,
# }

if os.name == "nt":
    # rc("text", usetex=True)  # I only have TeX on Windows :(
    os.chdir("D:/InteractingParticleSystems/noisysystem_temp")
elif os.name == "posix":
    os.chdir("/Volumes/Extreme SSD/InteractingParticleSystems/noisysystem_temp")

# Path to YAML file relative to current directory
# yaml_path = "../Experiments/one_cluster_low_gamma_ten_runs"
yaml_path = "./Experiments/OneClusterVaryGammaGlobalGarnierG"  # "./Experiments/one_cluster_vary_noise_scale_dt_100_runs_larger_gamma"
data_path = "Experiments/Data.nosync/"
history = get_main_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)
t, x, v = load_traj_data(file_names[0], data_path)
h = search_parameters["alpha"]


xi = 5 * np.sqrt((h - 4) / h)
ani = anim_torus(
    t.flatten(), x, v, mu_v=xi, subsample=200, variance=search_parameters["D"]
)
# variance=history[file_names[0]]["D"]
plt.show()

# #
# Writer = animation.writers["ffmpeg"]
# writer = Writer(fps=15, bitrate=-1)
# ani.save("metastability.mp4", writer=writer)
