import numpy as np
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import seaborn as sns

sns.set()
sns.color_palette("colorblind")


# def Q(x, phi, L=(2 * np.pi)):
#     """ Calculate order parameter of Wang et al. 2017"""
#     N = len(x)
#     for particle, position in enumerate(x):
#         distance = np.abs(x - position)
#         distance = (np.minimum(distance, L - distance))
#
#     return order_parameter
#
#
# def plot_order_parameter(x, phi, L):
#     """ Plot order parameter of Wang et al. 2017"""
#
#     return


def CL2(x, L=(2 * np.pi)):
    """Centered L2 discrepancy
    Adapted from https://stackoverflow.com/questions/50364048/
    python-removing-multiple-for-loops-for-faster-calculation-centered-l2-discrepa
    """
    N = len(x)
    term3 = 0
    term2 = np.sum(2.0 + np.abs(x / L - 0.5) - np.abs(x / L - 0.5) ** 2)
    for i in range(N):
        term3 += np.sum(
            1.0
            + np.abs(x[i] / L - 0.5) / 2
            + np.abs(x / L - 0.5) / 2
            - np.abs(x[i] / L - x / L) / 2
        )
    CL2 = (13 / 12) - (term2 - term3 / N) / N

    return CL2


def plot_avg_vel_CL2(avg_ax, cl2_ax, t, x, v, xi, ymax=None):
    # Plot average velocity and expected
    particle_count = len(x[0,])
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    avg_ax.plot(t, np.mean(v, axis=1))
    avg_ax.plot([0, t[-1]], [xi, xi], "--", c="orangered")
    avg_ax.plot([0, t[-1]], [-xi, -xi], "--", c="orangered")
    avg_ax.plot([0, t[-1]], [0, 0], "--", c="orangered")
    avg_ax.set(xlabel="Time", ylabel="Average Velocity", xlim=(0, t[-1]), ylim=(-4, 4))

    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = CL2(x[n,], L=10)

    cl2_ax.plot(t, CL2_vector)
    cl2_ax.plot([0, t[-1]], [exp_CL2, exp_CL2], "--")
    cl2_ax.set(xlabel="Time", ylabel="CL2", xlim=(0, t[-1]), ylim=(0, ymax))
    cl2_ax.ticklabel_format(axis="y", style="sci", scilimits=(-0, 1), useMathText=True)
    return avg_ax, cl2_ax


def anim_pos_vel_hist(
    t, _x, v, window=1, L=2 * np.pi, mu_v=1, variance=np.sqrt(2), framestep=1
):
    dt = t[1] - t[0]
    window //= dt
    x = (2 * np.pi / L) * _x  # Quick hack to rescale to circle.
    fig = plt.figure(figsize=(40, 10))
    fig.patch.set_alpha(0.0)
    grid = plt.GridSpec(1, 2, wspace=0.15, hspace=0.5)
    position_ax = plt.subplot(grid[0, 0])
    vel_ax = plt.subplot(grid[0, 1])
    fig.suptitle("t = {}".format(t[0]), fontsize=20)

    # Plotting vel histogram
    n_v, bins_v, patches_v = vel_ax.hist(
        v[0,],
        bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
        density=True,
        label="Velocity",
    )

    sigma = np.sqrt(variance)
    _v = np.arange(mu_v - 5 * sigma, mu_v + 5 * sigma, 0.01)
    pde_stationary_dist = stats.norm.pdf(_v, mu_v, sigma)

    vel_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_ax.set_ylim(0, pde_stationary_dist.max() + 0.05)
    # vel_ax.set_xlim(v.min(), v.max())
    vel_ax.set_xlim(0, 2)
    vel_ax.set_ylabel("Density", fontsize=15)

    # Plotting pos histogram
    n_x, bins_x, patches_x = position_ax.hist(
        x[0,],
        bins=np.arange(x.min(), x.max(), np.pi / 30),
        density=True,
        label="Position",
    )

    position_ax.set_ylim(0, 1.0)
    # position_ax.set_xlim(x.min(), x.max())
    position_ax.set_xlim(0, 2 * np.pi)

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

    position_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    position_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    position_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    mu_x = 1 / (2 * np.pi)

    _x = [x.min(), x.max()]
    position_ax.plot(_x, [mu_x, mu_x], label=r"Stationary D$^{\mathrm{n}}$")
    position_ax.set_ylabel("Density", fontsize=15)
    # fig.tight_layout()
    # fig.subplots_adjust(left=0.01, bottom=0.15)

    def animate(i, framestep):

        n_v, _ = np.histogram(
            v[max(0, i * framestep - window) : i * framestep,].flatten(),
            bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
            density=True,
        )
        n_x, _ = np.histogram(
            x[max(0, i * framestep - window) : i * framestep,],
            bins=np.arange(x.min(), x.max(), np.pi / 30),
            density=True,
        )
        # Update vel data
        for rect_v, height_v in zip(patches_v, n_v):
            rect_v.set_height(height_v)
        # Update pos data
        for rect_x, height_x in zip(patches_x, n_x):
            rect_x.set_height(height_x)

        fig.suptitle("t = {:.2f}".format(t[i * framestep]), fontsize=20)
        fig.show()

    ani = animation.FuncAnimation(
        fig, lambda i: animate(i, framestep), interval=60, frames=len(t) // framestep,
    )

    return ani


def anim_full(t, _x, v, L=2 * np.pi, mu_v=1, variance=np.sqrt(2), framestep=1):
    x = (2 * np.pi / L) * _x  # Quick hack to rescale to circle.
    fig = plt.figure(figsize=(40, 10))
    fig.patch.set_alpha(0.0)
    fig.text(
        0.7, 0.48, r"Position ($\theta$)", fontsize=15, ha="center", va="center",
    )
    fig.text(0.7, 0.05, r"Velocity", fontsize=15, ha="center", va="center")

    grid = plt.GridSpec(2, 4, wspace=0.15, hspace=0.5)

    torus_ax = plt.subplot(grid[0:2, 0:2])
    position_ax = plt.subplot(grid[0, 2])
    vel_ax = plt.subplot(grid[1, 2])

    position_time_ax = plt.subplot(grid[0, 3])
    vel_time_ax = plt.subplot(grid[1, 3])

    position_time_ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())
    vel_time_ax.yaxis.set_major_formatter(mpl.ticker.NullFormatter())

    an = np.linspace(0, 2 * np.pi, 100)
    torus_ax.plot(np.cos(an), np.sin(an), "-", alpha=0.5)
    torus_ax.axis("equal")
    fig.suptitle("t = {}".format(t[0]), fontsize=20)
    ### Plotting particles on torus #####
    torus_ax.set_ylim(-1.1, 1.1)
    torus_ax.set_xlim(-1.1, 1.1)
    torus_ax.set_facecolor("white")

    pos_vel = x[0, v[0,] >= 0]
    neg_vel = x[0, v[0,] < 0]
    # pos_vel = pos_vel[:50]  # only plot this many particles for clearer anim
    # neg_vel = neg_vel[:50]
    (neg_points,) = torus_ax.plot(
        np.sin(neg_vel),
        np.cos(neg_vel),
        linestyle="None",
        marker="o",
        alpha=0.5,
        ms=10,
    )
    (pos_points,) = torus_ax.plot(
        np.sin(pos_vel),
        np.cos(pos_vel),
        linestyle="None",
        marker="H",
        alpha=0.5,
        ms=10,
    )
    torus_ax.get_xaxis().set_visible(False)
    torus_ax.get_yaxis().set_visible(False)

    #########################################

    # Plotting vel histogram
    n_v, bins_v, patches_v = vel_ax.hist(
        v[0,],
        bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
        density=True,
        label="Velocity",
    )

    n_v_time, bins_v_time, patches_v_time = vel_time_ax.hist(
        v[0,],
        bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
        density=True,
        label="Velocity",
    )

    sigma = np.sqrt(variance)
    _v = np.arange(mu_v - 5 * sigma, mu_v + 5 * sigma, 0.01)
    pde_stationary_dist = stats.norm.pdf(_v, mu_v, sigma)

    vel_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_time_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_ax.set_ylim(0, pde_stationary_dist.max() + 0.05)
    # vel_ax.set_xlim(v.min(), v.max())
    vel_ax.set_xlim(0, 2)
    vel_ax.set_ylabel("Density", fontsize=15)
    vel_time_ax.set_ylim(0, pde_stationary_dist.max() + 0.05)
    vel_time_ax.set_xlim(v.min(), v.max())

    # Plotting pos histogram
    n_x, bins_x, patches_x = position_ax.hist(
        x[0,],
        bins=np.arange(x.min(), x.max(), np.pi / 30),
        density=True,
        label="Position",
    )
    n_x_time, bins_x_time, patches_x_time = position_time_ax.hist(
        x[0,],
        bins=np.arange(x.min(), x.max(), np.pi / 30),
        density=True,
        label="Position",
    )

    position_ax.set_ylim(0, 1.0)
    # position_ax.set_xlim(x.min(), x.max())
    position_ax.set_xlim(0, 2 * np.pi)
    position_time_ax.set_ylim(0, 1.0)
    # position_time_ax.set_xlim(x.min(), x.max())
    position_time_ax.set_xlim(0, 2 * np.pi)

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

    position_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    position_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    position_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))
    position_time_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    position_time_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    position_time_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    mu_v = 1 / (2 * np.pi)

    _x = [x.min(), x.max()]
    position_ax.plot(_x, [mu_v, mu_v], label=r"Stationary D$^{\mathrm{n}}$")
    position_time_ax.plot(_x, [mu_v, mu_v], label=r"Stationary D$^{\mathrm{n}}$")

    position_ax.set_ylabel("Density", fontsize=15)
    # fig.tight_layout()
    fig.subplots_adjust(left=0.01, bottom=0.15)

    def animate(i, framestep):
        # Particles
        pos_vel = x[i * framestep, v[i * framestep,] >= 0]
        neg_vel = x[i * framestep, v[i * framestep,] < 0]
        # pos_vel = pos_vel[:50]
        # neg_vel = neg_vel[:50]
        pos_points.set_data(np.sin(pos_vel), np.cos(pos_vel))
        neg_points.set_data(np.sin(neg_vel), np.cos(neg_vel))
        ####

        n_v, _ = np.histogram(
            v[i * framestep,].flatten(),
            bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
            density=True,
        )
        n_x, _ = np.histogram(
            x[i * framestep,],
            bins=np.arange(x.min(), x.max(), np.pi / 30),
            density=True,
        )

        n_v_time, _ = np.histogram(
            v[: i * framestep,].flatten(),
            bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
            density=True,
        )
        n_x_time, _ = np.histogram(
            x[: i * framestep,],
            bins=np.arange(x.min(), x.max(), np.pi / 30),
            density=True,
        )
        # Update vel data
        for rect_v, height_v in zip(patches_v, n_v):
            rect_v.set_height(height_v)
        for rect_v_time, height_v_time in zip(patches_v_time, n_v_time):
            rect_v_time.set_height(height_v_time)
        # Update pos data
        for rect_x, height_x in zip(patches_x, n_x):
            rect_x.set_height(height_x)
        for rect_x_time, height_x_time in zip(patches_x_time, n_x_time):
            rect_x_time.set_height(height_x_time)

        fig.suptitle("t = {:.2f}".format(t[i * framestep]), fontsize=20)
        fig.show()

    ani = animation.FuncAnimation(
        fig, lambda i: animate(i, framestep), interval=60, frames=len(t) // framestep,
    )

    return ani
