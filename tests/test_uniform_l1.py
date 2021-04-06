# import matplotlib.pyplot as plt
import numpy as np
from particle.statistics import calculate_l1_convergence

runs = 500
t = np.arange(0, 10, 0.5)
error_store = np.zeros(runs)
N = 480
for N in [480, 500, 600, 700, 1000]:
    for i in range(runs):
        x = np.random.uniform(low=0, high=2 * np.pi, size=N)
        # hist_x, bin_edges = np.histogram(
        #     x, bins=np.arange(0, 2 * np.pi, np.pi / 60), density=True
        # )
        # fig, ax = plt.subplots()
        # ax.plot(bin_edges[:-1], hist_x)
        # ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "--")
        # plt.show()
        x = np.tile(x, (len(t), len(x)))
        v = x
        error = calculate_l1_convergence(t, x, v)
        # print(np.mean(error))
        error_store[i] = np.mean(error)
        # plt.plot(t, error)
    print(f"Average error for {N} particles is {error_store.mean()}")
# plt.show()
