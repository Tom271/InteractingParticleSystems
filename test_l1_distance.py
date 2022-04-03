from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np


def corrected_calculate_l1_convergence(
    t,
    x,
    v,
    plot_hist: bool = False,
    final_plot_time: float = 100000,
    bin_count: int = 60,
):
    """Calculate l1 error between positions and uniform distribution

    Load data from file name and calculate the l1 discrepancy from a uniform
    distribution on the torus. Can also plot the histogram of the position
    density over time.
    """

    dt = t[1] - t[0]
    error = []
    if plot_hist is True:
        colormap = plt.get_cmap("viridis")
        fig, ax = plt.subplots()
        ax.set_prop_cycle(
            cycler(color=[colormap(k) for k in np.linspace(1, 0, int(1 / dt))])
        )
    # Go one step further as `np.arange` does not include the end point
    # while np.histogram expects the end point. Forces range to be [0,2\pi].
    bins = np.arange(0, 2 * np.pi * (1 + 1 / bin_count), step=(2 * np.pi / bin_count))
    for i in np.arange(0, int(min(len(t), final_plot_time // 0.5))):
        hist_x, bin_edges = np.histogram(x[i, :], bins=bins, density=True)

        # Multiply by width of the bin
        error_t = (2 * np.pi / bin_count) * np.abs((1 / (2 * np.pi)) - hist_x).sum()
        error.append(error_t)

        if plot_hist is True:
            ax.plot(bin_edges[:-1], hist_x)

    if plot_hist is True:
        ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "k--")
        ax.set(xlim=[0, 2 * np.pi], xlabel="Position", ylabel="Density")
        return error, fig, ax
    else:
        return error


# Test for range of bin counts and particle counts
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
            # Fake velocity data, unused
            v = x
            error = corrected_calculate_l1_convergence(t, x, v, bin_count=bin_count)

            error_store[k] = np.mean(error)

        particle_error_store[i, j] = error_store.mean()
        print(
            f"Average error for {N} particles, {bin_count} bins is {error_store.mean()}"
        )


fig, ax = plt.subplots()
im = ax.imshow(particle_error_store)
ax.set_xticks(np.arange(len(bins), step=2))
ax.set_xticklabels(bins[::2])
ax.set_yticks(np.arange(len(particle_counts), step=2))
ax.set_yticklabels(particle_counts[::2])
fig.colorbar(im)
ax.set_xlabel("Bin Count")
ax.set_ylabel("Particle Count")
plt.show()
