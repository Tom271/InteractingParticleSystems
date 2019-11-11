import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
import seaborn as sns
import pandas as pd
from particle import *

sns.set()
sns.color_palette("colorblind")
# CHANGE calculate_interaction BEFORE RUNNING -- SCALING BY DEFAULT IS N_i NOT TOTAL PARTICLES


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


def plot_figure_2():
    print("Plotting Figure 2...")
    fig2, ax = plt.subplots(2, 2, figsize=(12.0, 12.0))

    # Figure 2
    dt = 0.1
    particle_count = 500
    diffusion = (2 ** 2) / 2
    T_final = 100
    length = 10
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)

    # Fig 2(a-b)
    well_depth = 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    xi = 1
    plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=5e-3)

    # Fig 2(c-d)
    well_depth = 6
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)

    plot_avg_vel_CL2(ax[1, 0], ax[1, 1], t, x, v, xi, ymax=3.5e-3)

    fig2.suptitle("Garnier Fig 2, Vary Well Depth")
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.9)
    fig2.savefig("Figure2.jpg", format="jpg", dpi=250)
    # plt.show()


# Figure 3
def plot_figure_3():
    print("Plotting Figure 3...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 6
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    length = 10
    fig, ax = plt.subplots(3, 2, figsize=(18.0, 12.0))

    # Figure 3(a-b)
    diffusion = (0.5 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=0.05)

    # Figure 3(c-d)
    diffusion = (1.0 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[1, 0], ax[1, 1], t, x, v, xi, ymax=8e-4)
    # Figure 3(e-f)
    diffusion = (1.5 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[2, 0], ax[2, 1], t, x, v, xi, ymax=6e-4)
    fig.suptitle("Garnier Fig 3, Vary Diffusion")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("Figure3.jpg", format="jpg", dpi=250)
    # plt.show()


def plot_figure_4():
    print("Plotting Figure 4...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    well_depth = 6
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    diffusion = (0.5 ** 2) / 2
    length = 10

    fig, ax = plt.subplots(2, 2, figsize=(12.0, 12.0))
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # USe pd dataframe to get all in one plot
    fig1, ax1 = plt.subplots()
    print("...slowly ...")
    ax1 = sns.kdeplot(
        np.repeat(t[: int(20 // dt)], particle_count),
        x[: int(20 // dt),].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax1.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(0, 20),
        ylim=(0, length),
        title="First 20s KDE",
    )
    # TODO: Change marker size and shape, maybe swap for plot
    ax[0, 1].scatter(
        np.repeat(t[: int(20 // dt), np.newaxis], 50, axis=1),
        x[: int(20 // dt), 0:50],
        marker=".",
        s=1,
    )
    ax[0, 1].set(
        xlabel="Time",
        ylabel="Position",
        xlim=(0, 20),
        ylim=(0, length),
        title="First 20s Traj",
    )

    fig2, ax2 = plt.subplots()
    ax2 = sns.kdeplot(
        np.repeat(t[-int(20 // dt) :], particle_count),
        x[-int(20 // dt) :,].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax2.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(T_final - 20, T_final),
        ylim=(0, length),
        title="Last 20s KDE",
    )
    # TODO: Change marker size and shape
    ax[1, 1].scatter(
        np.repeat(t[-int(20 // dt) :, np.newaxis], 50, axis=1),
        x[-int(20 // dt) :, 0:50],
        marker=".",
        s=1,
    )
    ax[1, 1].set(
        xlabel="Time",
        ylabel="Position",
        xlim=(80, 100),
        ylim=(0, length),
        title="Last 20s Traj",
    )
    fig.suptitle("Garnier Fig 4,Cluster at Low Noise")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("Figure4.jpg", format="jpg", dpi=250)

    fig1.savefig("Figure4b.jpg", format="jpg", dpi=250)
    fig2.savefig("Figure4d.jpg", format="jpg", dpi=250)
    # plt.show()


def plot_figure_5():
    print("Plotting Figure 5...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 6
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    length = 10
    fig, ax = plt.subplots(3, 2, figsize=(18.0, 12.0))

    # Figure 5(a-b)
    diffusion = (0.5 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=0.07)

    # Figure 5(c-d)
    diffusion = (1.0 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[1, 0], ax[1, 1], t, x, v, xi, ymax=0.05)
    # Figure 5(e-f)
    diffusion = (1.5 ** 2) / 2

    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[2, 0], ax[2, 1], t, x, v, xi, ymax=0.012)
    fig.suptitle("Garnier Fig 5, Vary Diffusion, 0 Start")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("Figure5.jpg", format="jpg", dpi=250)
    # plt.show()


def plot_figure_6():
    print("Plotting Figure 6...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 10
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    diffusion = (1 ** 2) / 2
    length = 10
    fig, ax = plt.subplots(2, 2, figsize=(18.0, 12.0))

    # Figure 6(a-b)
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=0.025)

    fig2, ax2 = plt.subplots()
    ax2 = sns.kdeplot(
        np.repeat(t[: int(20 // dt)], particle_count),
        x[: int(20 // dt),].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax2.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(0, 20),
        ylim=(0, length),
        title="First 20s KDE",
    )

    fig3, ax3 = plt.subplots()
    ax3 = sns.kdeplot(
        np.repeat(t[-int(20 // dt) :], particle_count),
        x[-int(20 // dt) :,].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax3.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(T_final - 20, T_final),
        ylim=(0, length),
        title="Last 20s KDE",
    )
    fig.savefig("Figure6.jpg", format="jpg", dpi=250)
    fig2.savefig("Figure6b.jpg", format="jpg", dpi=250)
    fig3.savefig("Figure6d.jpg", format="jpg", dpi=250)
    # plt.show()


def plot_figure_7():
    print("Plotting Figure 7...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 5
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    diffusion = (1 ** 2) / 2
    length = 10
    fig, ax = plt.subplots(2, 2, figsize=(18.0, 12.0))

    # Figure 7(a-b)
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Smooth",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    # Plot average velocity and expected
    plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=0.025)

    fig2, ax2 = plt.subplots()
    ax2 = sns.kdeplot(
        np.repeat(t[: int(20 // dt)], particle_count),
        x[: int(20 // dt),].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax2.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(0, 20),
        ylim=(0, length),
        title="First 20s KDE",
    )

    fig3, ax3 = plt.subplots()
    ax3 = sns.kdeplot(
        np.repeat(t[-int(20 // dt) :], particle_count),
        x[-int(20 // dt) :,].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax3.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(T_final - 20, T_final),
        ylim=(0, length),
        title="Last 20s KDE",
    )
    fig.savefig("Figure7.jpg", format="jpg", dpi=250)
    fig2.savefig("Figure7b.jpg", format="jpg", dpi=250)
    fig3.savefig("Figure7d.jpg", format="jpg", dpi=250)
    # plt.show()


if __name__ == "__main__":
    # NOT USING GARNIER SCALING ARE YOU?
    # Using smooth interaction functuion
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    plot_figure_6()
    plot_figure_7()
