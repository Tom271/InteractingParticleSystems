import numpy as np
import scipy.stats as stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from particle.simulate import CL2

sns.set()
sns.color_palette("colorblind")


def plot_avg_vel_CL2(avg_ax, cl2_ax, t, x, v, xi, ymax=None):
    """ Plot average velocity and centered L2 discrepancy

    Produces a figure showing average velocity and CL2 discrepancy. Calculates CL2,
    so can be slow

    Args:
        avg_ax: Axes object on which to plot avg velocity
        cl2_ax: Axes object on which to plot calculated CL2 discrepancy
        t: Time data, array
        x: Position data , array
        v: Velocity data, array
        xi: Solution to compatibility equation (usually 1 or -1)
        ymax: Optional, enforces y axis limit on CL2 plot

    Returns:
        avg_ax: Axes object with plot
        cl2_ax: Axes object with plot
    """
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
    """ Animate position and velocity histograms

    Produces an animation object with histograms of positions and velocities of
    particles across a window.

    Args:
        t: Time data, 1d array
        _x: Particle position data, array
        v: Particle velocity data, array
        window: Time across which the density should be approximated, between 0 and T,
                float
        L: Length of the domain
        mu_v: Expected mean of the velocity stationary distribution, float
        variance: Expected variance of the velocity stationary distribution, float (>0)
        framestep: The number of frames the animation should jump,
                   integer greater than 1

    Returns:
        ani: Animation object
    """
    dt = t[1] - t[0]
    window //= dt
    x = (2 * np.pi / L) * _x  # Quick hack to rescale to circle.
    fig = plt.figure(figsize=(16, 4))
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
    position_ax.set_xlim(0, 2 * np.pi)

    # Helper to label axes using pi symbol

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


def anim_torus(
    t,
    _x,
    v,
    L=2 * np.pi,
    mu_v=1,
    variance=np.sqrt(2),
    framestep=1,
    pos_panel=None,
    vel_panel=None,
    subsample=None,
):
    """ Animate the particles on the torus

    Produces an animation of the particles moving on the torus, as well as two panels.
    One contains two plots using position data, the other using the velocity data.
    Panels can be either histograms (density estimates) or particle trajectories.

    Args:
        t:
        _x:
        v:
        L:
        mu_v:
        variance:
        framestep:
        pos_panel:
        vel_panel:
        subsample:

    Returns:
        ani:

    See also: update_torus, plot_pos_hist, plot_vel_hist, update_pos_hist,
    update_pos_line, update_vel_hist, update_vel_line
    """
    x = (2 * np.pi / L) * _x  # Quick hack to rescale to circle.
    fig = plt.figure(figsize=(16, 4))
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
    # Plotting particles on torus
    torus_ax.set_ylim(-1.1, 1.1)
    torus_ax.set_xlim(-1.1, 1.1)
    torus_ax.set_facecolor("white")

    if subsample:
        # Take a random sample of points to plot on torus
        subsample_mask = np.random.choice(len(x[0,]), size=subsample, replace=False)
        x_subsample = x[:, subsample_mask]
        v_subsample = v[:, subsample_mask]

        pos_vel = x_subsample[0, v_subsample[0,] >= 0]
        neg_vel = x_subsample[0, v_subsample[0,] < 0]
    else:
        pos_vel = x[0, v[0,] >= 0]
        neg_vel = x[0, v[0,] < 0]

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
    if pos_panel == "line":
        pos_lines = plot_pos_line(position_ax, position_time_ax, t, x)
    else:
        (
            n_x,
            bins_x,
            patches_x,
            n_x_time,
            bins_x_time,
            patches_x_time,
        ) = plot_pos_hist(position_ax, position_time_ax, x)

    if vel_panel == "line":
        vel_lines = plot_vel_line(vel_ax, vel_time_ax, t, v)
    else:
        (
            n_v,
            bins_v,
            patches_v,
            n_v_time,
            bins_v_time,
            patches_v_time,
        ) = plot_vel_hist(vel_ax, vel_time_ax, v, mu_v, variance)

    # fig.tight_layout()
    fig.subplots_adjust(left=0.01, bottom=0.15)

    def animate(i, framestep):
        if subsample:
            update_torus(
                i, t, x_subsample, v_subsample, framestep, pos_points, neg_points,
            )
        else:
            update_torus(i, t, x, v, framestep, pos_points, neg_points)

        # update_pos_hist()  update_pos_line(subsample)
        if pos_panel == "line":
            update_pos_line(i, t, x, framestep, pos_lines)

        else:
            update_pos_hist(i, x, framestep, patches_x, patches_x_time)

        # update_vel_hist() or update_vel_line(subsample)
        if vel_panel == "line":
            update_vel_line(i, t, v, framestep, vel_lines)

        else:
            update_vel_hist(i, v, framestep, patches_v, patches_v_time)

        fig.suptitle("t = {:.2f}".format(t[i * framestep]), fontsize=20)
        fig.show()

    ani = animation.FuncAnimation(
        fig, lambda i: animate(i, framestep), interval=60, frames=len(t) // framestep,
    )

    return ani


def plot_vel_hist(vel_ax, vel_time_ax, v, mu_v, variance):
    """Plotting vel histogram"""
    n_v, bins_v, patches_v = vel_ax.hist(
        v[0,],
        bins=np.arange(
            min(-2, v.min()), max(2, v.max()), max(0.1, (v.max() - v.min()) / 30),
        ),
        density=True,
        label="Velocity",
    )

    n_v_time, bins_v_time, patches_v_time = vel_time_ax.hist(
        v[0,],
        bins=np.arange(
            min(-2, v.min()), max(2, v.max()), max(0.1, (v.max() - v.min()) / 30),
        ),
        density=True,
        label="Velocity",
    )

    sigma = np.sqrt(variance)
    _v = np.arange(mu_v - 5 * sigma, mu_v + 5 * sigma, 0.01)
    pde_stationary_dist = stats.norm.pdf(_v, mu_v, sigma)

    vel_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_time_ax.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    vel_ax.set_ylim(0, pde_stationary_dist.max() + 0.05)
    vel_ax.set_xlim(min(-2, v.min()), max(2, v.max()))

    vel_ax.set_ylabel("Density", fontsize=15)
    vel_time_ax.set_ylim(0, pde_stationary_dist.max() + 0.05)
    vel_time_ax.set_xlim(min(-2, v.min()), max(2, v.max()))

    return n_v, bins_v, patches_v, n_v_time, bins_v_time, patches_v_time


def plot_pos_hist(position_ax, position_time_ax, x):
    """ Plotting pos histogram"""
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
    position_ax.set_xlim(0, 2 * np.pi)
    position_time_ax.set_ylim(0, 1.0)
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

    # PLot stationary distribution
    mu_x = 1 / (2 * np.pi)
    _x = [x.min(), x.max()]
    position_ax.plot(_x, [mu_x, mu_x], label=r"Stationary D$^{\mathrm{n}}$")
    position_time_ax.plot(_x, [mu_x, mu_x], label=r"Stationary D$^{\mathrm{n}}$")

    position_ax.set_ylabel("Density", fontsize=15)

    return n_x, bins_x, patches_x, n_x_time, bins_x_time, patches_x_time


def plot_pos_line(position_ax, position_time_ax, t, x):
    """Plots the position trajectories"""
    pos_lines = []
    for index in range(len(x[0,])):
        lobj = position_ax.plot([], [], lw=1, alpha=0.1)[0]
        pos_lines.append(lobj)
    avg_line = position_ax.plot([], [], lw=2, alpha=1)[0]
    pos_lines.append(avg_line)

    position_ax.set_ylim(0, 2 * np.pi)
    position_time_ax.set_ylim(0, 2 * np.pi)
    position_ax.set_xlim(0, t[-1])
    position_time_ax.set_xlim(0, t[-1])
    return pos_lines


def plot_vel_line(vel_ax, vel_time_ax, t, v):
    """Plots the velocity trajectories """
    vel_lines = []
    for index in range(len(v[0,])):
        lobj = vel_ax.plot([], [], lw=1, alpha=0.1)[0]
        vel_lines.append(lobj)
    avg_line = vel_ax.plot([], [], lw=2, alpha=1)[0]
    vel_lines.append(avg_line)

    vel_ax.set_ylim(min(-1.1, v.min()), max(1.1, v.max()))
    vel_time_ax.set_ylim(-1.1, v.min()), max(1.1, v.max())
    vel_ax.set_xlim(0, t[-1])
    vel_time_ax.set_xlim(0, t[-1])
    return vel_lines


def update_torus(i, t, x, v, framestep, pos_points, neg_points):
    """ Update particles positions on torus plot  """
    # Update positions on torus
    pos_vel = x[i * framestep, v[i * framestep,] >= 0]
    neg_vel = x[i * framestep, v[i * framestep,] < 0]

    pos_points.set_data(np.sin(pos_vel), np.cos(pos_vel))
    neg_points.set_data(np.sin(neg_vel), np.cos(neg_vel))


def update_pos_hist(i, x, framestep, patches_x, patches_x_time):
    """ Update position histograms """
    n_x, _ = np.histogram(
        x[i * framestep,], bins=np.arange(x.min(), x.max(), np.pi / 30), density=True,
    )

    n_x_time, _ = np.histogram(
        x[: i * framestep,], bins=np.arange(x.min(), x.max(), np.pi / 30), density=True,
    )

    # Update pos data
    for rect_x, height_x in zip(patches_x, n_x):
        rect_x.set_height(height_x)
    for rect_x_time, height_x_time in zip(patches_x_time, n_x_time):
        rect_x_time.set_height(height_x_time)


def update_vel_hist(i, v, framestep, patches_v, patches_v_time):
    """ Update velocity histograms """
    n_v, _ = np.histogram(
        v[i * framestep,].flatten(),
        bins=np.arange(
            min(-2, v.min()), max(2, v.max()), max(0.1, (v.max() - v.min()) / 30),
        ),
        density=True,
    )
    n_v_time, _ = np.histogram(
        v[: i * framestep,].flatten(),
        bins=np.arange(
            min(-2, v.min()), max(2, v.max()), max(0.1, (v.max() - v.min()) / 30),
        ),
        density=True,
    )
    # Update vel data
    for rect_v, height_v in zip(patches_v, n_v):
        rect_v.set_height(height_v)
    for rect_v_time, height_v_time in zip(patches_v_time, n_v_time):
        rect_v_time.set_height(height_v_time)
    return


def update_pos_line(i, t, x, framestep, pos_lines):
    """ Update position trajectories """
    for lnum, line in enumerate(pos_lines[:-1]):
        line.set_data(
            t[: i * framestep], x[: i * framestep, lnum]
        )  # set data for each line separately.
    pos_lines[-1].set_data(t[: i * framestep], np.mean(x[: i * framestep,], axis=1))


def update_vel_line(i, t, v, framestep, vel_lines):
    """ Update velocity trajectories """
    for lnum, line in enumerate(vel_lines[:-1]):
        line.set_data(
            t[: i * framestep], v[: i * framestep, lnum]
        )  # set data for each line separately.
    vel_lines[-1].set_data(t[: i * framestep], np.mean(v[: i * framestep,], axis=1))


if __name__ == "__main__":
    import os
    import pickle

    cur_path = os.path.dirname(__file__)
    test_data_path = os.path.relpath("..\\tests\\plot_test_data", cur_path)
    # test_data_path = os.path.relpath(
    #     "..\\Experiments\\1712Data\\ZeroNoiseDetIC_v_take2", cur_path
    # )
    test_data = pickle.load(open(test_data_path, "rb"))
    t = test_data["Time"]
    x = test_data["Position"]
    v = test_data["Velocity"]
    annie = anim_torus(
        t,
        x,
        v,
        mu_v=-1,
        variance=1,
        L=2 * np.pi,
        framestep=1,
        # pos_panel="line",
        # vel_panel="line",
    )
    plt.show()
