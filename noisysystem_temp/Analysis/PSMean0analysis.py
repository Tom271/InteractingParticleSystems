from matplotlib import rc
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from particle.processing import get_master_yaml, match_parameters, load_traj_data


rc("text", usetex=True)
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
history = get_master_yaml(yaml_path)

# fig, ax = plt.subplots(1, 1, figsize=(15, 5), sharex=True)
cm = plt.get_cmap("coolwarm")
cNorm = colors.DivergingNorm(vmin=0.01, vcenter=0.05, vmax=0.25)
scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
file_names = match_parameters(search_parameters, history)
for idx, file_name in enumerate(file_names):
    print(file_name)
    # t, error = calculate_l1_convergence(file_name, plot_hist=False)
    t, x, v = load_traj_data(file_name)
    avg_vel = v.mean(axis=1)
    if idx == 0:
        avg_vel_store = np.zeros((len(file_names), len(avg_vel)))
    avg_vel_store[idx, :] = avg_vel


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
for i in range(3):
    plt.plot(t, avg_vel_store[i, :], alpha=0.15)
g.ax_marg_x.set_axis_off()
sns.kdeplot(avg_vel_store[-100, :], ax=g.ax_marg_y, legend=False, vertical=True)
# g.ax_marg_y.plot([0,1], [1,1])
# g.ax_marg_y.plot([0,1], [-1,-1])

plt.plot([0, t[-1]], [1, 1], "k--", alpha=0.12)
plt.plot([0, t[-1]], [-1, -1], "k--", alpha=0.12)
plt.xlabel("Time")
plt.ylabel(r"$\bar{M}^N(t)$")
# ax1.set(xlabel="Time", ylabel=r"$\ell^1$ Error")
# .set(xlabel="Time", ylabel=r"$M^N(t)$")
plt.tight_layout()
plt.show()

# fig.savefig(f"OneClusterVaryGamma.jpg", dpi=300)
