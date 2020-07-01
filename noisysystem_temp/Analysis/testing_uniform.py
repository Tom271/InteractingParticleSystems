from analysis_helper import numba_hist
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# Standard plotting choices
rc("text", usetex=True)
sns.set(style="white", context="talk")

time = 200
particle_counts = np.concatenate((np.logspace(0, 6, num=10, dtype=int), [480]))
mean_error = []
fig, ax = plt.subplots()
particle_counts = sorted(particle_counts)
for particles in particle_counts:
    # particles = 10**6
    uniform_points = np.random.default_rng().uniform(0, 2 * np.pi, particles * time)
    uniform_points = uniform_points.reshape((time, particles))
    error = []
    for i in range(time):
        hist_x, bin_edges = numba_hist(uniform_points, i)
        error_t = np.abs(1 / (2 * np.pi) - hist_x).sum()
        error.append(error_t)
    mean_error.append(np.mean(error))
    print(mean_error[-1])
    if particles == 480:
        ax.annotate(
            rf"(N=480, error = {mean_error[-1]:.3})",
            xy=(480, mean_error[-1]),
            textcoords="data",
        )
ax.loglog(particle_counts, mean_error)
ax.plot([480, 480], [min(mean_error), max(mean_error)], "--", alpha=0.5)
ax.set(xlabel="Particles", ylabel=r"$\ell^1$ Error")
plt.tight_layout()
plt.show()
