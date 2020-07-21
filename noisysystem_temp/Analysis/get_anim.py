from matplotlib import rc
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import seaborn as sns

from particle.plotting import anim_torus
from particle.processing import get_master_yaml, match_parameters, load_traj_data

# Standard plotting choices
rc("text", usetex=True)
sns.set(style="white", context="talk")

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

# Path to YAML file relative to current directory
yaml_path = "../Experiments/one_cluster_low_gamma_ten_runs"
history = get_master_yaml(yaml_path)
file_names = match_parameters(search_parameters, history)
t, x, v = load_traj_data(file_names[0])

ani = anim_torus(t.flatten(), x, v, subsample=100, variance=history[file_names[0]]["D"])
plt.show()


Writer = animation.writers["ffmpeg"]
writer = Writer(fps=15, bitrate=-1)
# ani.save('test.mp4', writer=writer)
