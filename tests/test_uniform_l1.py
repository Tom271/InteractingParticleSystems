import matplotlib.pyplot as plt
import numpy as np
from particle.statistics import corrected_calculate_l1_convergence

runs = 5
t = np.arange(0, 20, 1)
error_store = np.zeros(runs)
rng = np.random.default_rng()

particle_counts = np.geomspace(10000, 10, 50, dtype=int)
bins = np.geomspace(10, 200, 50, dtype=int)
particle_error_store = np.zeros((len(particle_counts), len(bins)))

for i, N in enumerate(particle_counts):
    for j, bin_count in enumerate(bins):
        for k in range(runs):
            x = rng.uniform(low=0, high=2 * np.pi, size=(len(t), N))
            # for j in range(len(t)):
            #     hist_x, bin_edges = np.histogram(x[j, :], bins=60, density=True)
            # fig, ax = plt.subplots()
            # ax.plot(bin_edges[:-1], hist_x)
            # ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "--")
            # plt.show()
            # x = np.tile(x, (len(t), len(x)))
            v = x
            error = corrected_calculate_l1_convergence(t, x, v, bin_count=bin_count)
            # plt.show()
            # print(np.mean(error))
            error_store[k] = np.mean(error)

            # plt.plot(t, error)
        particle_error_store[i, j] = error_store.mean()
        print(
            f"Average error for {N} particles, {bin_count} bins is {error_store.mean()}"
        )

fig, ax = plt.subplots()
im = ax.imshow(particle_error_store)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(bins), step=2))
ax.set_xticklabels(bins[::2])
ax.set_yticks(np.arange(len(particle_counts), step=2))
ax.set_yticklabels(particle_counts[::2])
fig.colorbar(im)
ax.set_xlabel("Bin Count")
ax.set_ylabel("Particle Count")
plt.show()
#
# N = int(480)
# x = rng.uniform(low=0, high=2 * np.pi, size=N)
# bin_count = 60
# # error = corrected_calculate_l1_convergence(t, x, v).
# # np.linspace(0, 2*np.pi, num=2)
# bins = np.append(np.arange(0, 2 * np.pi, step=(2 * np.pi / bin_count)), [2 * np.pi])
# hist_x, bin_edges = np.histogram(x, bins=bins, density=True)
# error = (2 * np.pi / bin_count) * np.abs((1 / (2 * np.pi)) - hist_x).sum()
#
# # plt.plot(t, error)
# print(f"Average error for {N} particles is {error.mean()}")
# fig, ax = plt.subplots()
#
# ax.bar(bin_edges[:-1], hist_x, width=(2 * np.pi / (bin_count)), align="edge")
# ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "k--")
# ax.set_ylim([0, 0.2])
# plt.show()
