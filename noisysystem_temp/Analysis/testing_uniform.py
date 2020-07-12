import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import rc

# Standard plotting choices
# rc("text", usetex=True)
sns.set(style="white", context="talk")

time = 200
particle_counts = np.concatenate((np.logspace(0, 5, num=10, dtype=int), [100]))
mean_error = []
fig, ax = plt.subplots()
particle_counts = sorted(particle_counts)
for particles in particle_counts:
    # particles = 10**6
    uniform_points = np.random.default_rng().uniform(0, 2 * np.pi, particles * time)
    uniform_points = uniform_points.reshape((time, particles))
    error = []
    for i in range(time):
        hist_x, bin_edges = np.histogram(
            uniform_points[i, :], bins=np.arange(0, 2 * np.pi, np.pi / 60), density=True
        )
        error_t = np.abs(1 / (2 * np.pi) - hist_x).sum()
        error.append(error_t)
    mean_error.append(np.mean(error))
    print(mean_error[-1])
    if particles == 100:
        ax.annotate(
            rf"(N=100, error = {mean_error[-1]:.3})",
            xy=(100, mean_error[-1]),
            textcoords="data",
        )
ax.loglog(particle_counts, mean_error)
ax.plot([100, 100], [min(mean_error), max(mean_error)], "--", alpha=0.5)
ax.set(xlabel="Particles", ylabel=r"$\ell^1$ Error")
plt.tight_layout()
plt.show()
