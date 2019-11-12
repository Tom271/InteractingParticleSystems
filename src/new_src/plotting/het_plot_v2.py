import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns


# import pickle

sns.set()
sns.color_palette("colorblind")


def anim_full(t, _x, v, L=2 * np.pi, mu=1, variance=np.sqrt(2), framestep=1):
    x = (2 * np.pi / L) * _x  # Quick hack to rescale to circle.
    fig = plt.figure(figsize=(20, 10))
    fig.patch.set_alpha(0.0)

    grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)

    big_ax = plt.subplot(grid[0:2, 0:2])
    pos_ax = plt.subplot(grid[0, 2])
    vel_ax = plt.subplot(grid[1, 2])

    an = np.linspace(0, 2 * np.pi, 100)
    big_ax.plot(np.cos(an), np.sin(an), "--", alpha=0.5)
    big_ax.axis("equal")
    fig.suptitle("t = {}".format(t[0]), fontsize=25)
    ### Plotting particles on torus #####
    big_ax.set_ylim(-1.1, 1.1)
    big_ax.set_xlim(-1.1, 1.1)

    # Horrible trick to plot different colours -- look for way to pass color argument in one go
    # pos_vel = [x[0, idx] if vel >= 0 else None for idx, vel in enumerate(v[0, :])]
    # pos_vel = np.array([x for x in pos_vel if x is not None])
    # neg_vel = [x[0, idx] if vel <= 0 else None for idx, vel in enumerate(v[0, :])]
    # neg_vel = [x for x in neg_vel if x is not None]
    # x and y wrong way round so that +ve vel is clockwise

    pos_vel = x[0, v[0,] >= 0]
    neg_vel = x[0, v[0,] < 0]
    (neg_points,) = big_ax.plot(
        np.sin(neg_vel), np.cos(neg_vel), linestyle="None", marker="o", alpha=0.5, ms=10
    )
    (pos_points,) = big_ax.plot(
        np.sin(pos_vel), np.cos(pos_vel), linestyle="None", marker="o", alpha=0.5, ms=8
    )

    big_ax.tick_params(axis="both", which="major", labelsize=20)
    big_ax.set_xlabel("x", fontsize=25)
    big_ax.set_ylabel("y", fontsize=25)
    #########################################

    # Plotting vel histogram
    n_v, bins_v, patches_v = vel_ax.hist(
        v[0, :],
        bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
        density=True,
        label="Velocity",
    )

    well_depth = 5
    # 5*np.sqrt((well_depth-4)/well_depth)#np.sign(np.mean(v[0,:]))
    sigma = np.sqrt(variance)
    _v = np.arange(mu - 5 * sigma, mu + 5 * sigma, 0.01)
    pde_stationary_dist = stats.norm.pdf(_v, mu, sigma)

    vel_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_ax.set_ylim(0, pde_stationary_dist.max())
    vel_ax.set_xlim(v.min(), v.max())
    vel_ax.set_xlabel("Velocity", fontsize=20)
    vel_ax.set_ylabel("Density", fontsize=20)

    # PLotting pos histogram
    n_x, bins_x, patches_x = pos_ax.hist(
        x[0, :], bins=np.arange(x.min(), x.max(), 0.15), density=True, label="Position"
    )

    pos_ax.set_ylim(0, 1.05)
    pos_ax.set_xlim(x.min(), x.max())

    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    pos_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    pos_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    pos_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    mu = 1 / L  # 1/(2*np.pi)

    _x = [x.min(), x.max()]
    pos_ax.plot(_x, [mu, mu], label=r"Stationary D$^{\mathrm{n}}$")

    pos_ax.set_xlabel(r"Position ($\theta$)", fontsize=20)
    pos_ax.set_ylabel("Density", fontsize=20)

    def animate(i, framestep):
        # Particles
        pos_vel = [
            x[i * framestep, idx] if vel >= 0 else None
            for idx, vel in enumerate(v[i * framestep,])
        ]
        pos_vel = [x for x in pos_vel if x is not None]
        neg_vel = [
            x[i * framestep, idx] if vel <= 0 else None
            for idx, vel in enumerate(v[i * framestep,])
        ]
        neg_vel = [x for x in neg_vel if x is not None]
        pos_points.set_data(np.sin(pos_vel), np.cos(pos_vel))
        neg_points.set_data(np.sin(neg_vel), np.cos(neg_vel))
        ####

        n_v, _ = np.histogram(
            v[: i * framestep, :].flatten(),
            bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
            density=True,
        )
        n_x, _ = np.histogram(
            x[: i * framestep, :], bins=np.arange(x.min(), x.max(), 0.15), density=True
        )

        # Update vel data
        for rect_v, height_v in zip(patches_v, n_v):
            rect_v.set_height(height_v)
        # Update pos data
        for rect_x, height_x in zip(patches_x, n_x):
            rect_x.set_height(height_x)

        fig.suptitle("t = {:.2f}".format(t[i * framestep]), fontsize=25)
        fig.show()

    ani = animation.FuncAnimation(
        fig, lambda i: animate(i, framestep), interval=60, frames=len(t) // framestep
    )

    return ani
