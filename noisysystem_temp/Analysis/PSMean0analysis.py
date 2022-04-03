import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import rc as rc
import seaborn as sns
import os

from particle.processing import get_main_yaml, match_parameters, load_traj_data


# rc("text", usetex=True)
sns.set(style="white", context="talk")

search_parameters = {
    "particle_count": 100,
    "G": "Smooth",
    "scaling": "Local",
    "D": 1.0,
    "phi": "Gamma",
    "gamma": 0.1,
    "initial_dist_x": "uniform_dn",
    "initial_dist_v": "pos_normal_dn",
    "T_end": 2000.0,
    "dt": 0.01,
    "option": "numba",
}

yaml_path = "../Experiments/PSMean0"
history = get_main_yaml(yaml_path)
rc("text", usetex=True)
# fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
cm = plt.get_cmap("coolwarm")
cNorm = colors.TwoSlopeNorm(vmin=0.01, vcenter=0.05, vmax=0.25)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
file_names = match_parameters(search_parameters, history)
os.chdir("D:/InteractingParticleSystems/noisysystem_temp/")
# Path to YAML file relative to current directory
yaml_path = "./Experiments/one_cluster_vary_gamma_50_runs_higher_particles"

for idx, file_name in enumerate(file_names):
    print(file_name)
    # t, error = calculate_l1_convergence(file_name, plot_hist=False)
    t, x, v = load_traj_data(file_name, data_path="Experiments/parquet_data/")
    avg_vel = v.mean(axis=1)
    if idx == 0:
        avg_vel_store = np.zeros((len(file_names), len(avg_vel)))
    avg_vel_store[idx, :] = avg_vel

# fig, ax = plt.subplots()
# ax.plot(
#     t,
#     np.mean(avg_vel_store, axis=0),
#     color="k",
#     label=f"{history[file_name]['gamma']}",
#     alpha=0.8,
#     zorder=2,
# )
grid = sns.JointGrid(x=t.flatten(), y=np.mean(avg_vel_store, axis=0))

g = grid.plot_joint(sns.lineplot)
for i in range(2):
    g.ax_joint.plot(
        t, avg_vel_store[4 * i, :], alpha=0.15
    )  # randomly chosen for a switch
g.ax_marg_x.set_axis_off()
sns.kdeplot(y=avg_vel_store[-100, :], ax=g.ax_marg_y, legend=False)
# g.ax_marg_y.plot([0,1], [1,1])
# g.ax_marg_y.plot([0,1], [-1,-1])

g.ax_joint.plot([0, t[-1]], [1, 1], "k", alpha=0.12)
g.ax_joint.plot([0, t[-1]], [-1, -1], "k", alpha=0.12)
g.set_axis_labels(xlabel="Time", ylabel=r"$\bar{M}^N(t)$")
# g.ax_joint.ylabel()
# ax1.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
# .set(xlabel="Time", ylabel=r"$M^N(t)$")
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryGamma.jpg", dpi=300)
