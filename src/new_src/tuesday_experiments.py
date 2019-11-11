import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np
import seaborn as sns
from particle import *
import GarnierFigures as GF

"""Experiments from discussion on 04/11
See pg 83 of notebook.
1) Repeating cluster at 0.1 in Fig3 (Garnier)
2) Making clearer animations with fewer particles
3) Running fewer particles for longer
4) Taking time averages to smooth periodicity
5) Repeating small well sim
6) Trying to break clusters with small range phi

"""
# Experiment 1


def repeat_cluster_formation(trials=1):
    print("Running Exp 1i)...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 6
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    length = 10
    cl2_fig, cl2_ax = plt.subplots(figsize=(10.0, 5.0))
    v_fig, v_ax = plt.subplots(figsize=(10.0, 5.0))
    diffusion = (1 ** 2) / 2
    mean_v = 0
    mean_x = 0
    mean_cl2 = 0
    for _ in range(trials):
        t, x, v = run_full_particle_system(
            particles=particle_count,
            dt=dt,
            initial_dist_v=normal(
                loc=xi, scale=np.sqrt(diffusion), size=particle_count
            ),
            D=diffusion,
            interaction_function="Garnier",
            herding_function="Garnier",
            T_end=T_final,
            L=length,
            well_depth=well_depth,
        )
        # Plot average velocity and expected
        CL2_vector = np.zeros(len(t))
        for n in range(len(t)):
            CL2_vector[n] = CL2(x[n,], L=10)

        cl2_ax.plot(t, CL2_vector, c="gray", alpha=0.3)
        v_ax.plot(t, np.mean(v, axis=1), c="gray", alpha=0.3)
        mean_v += v
        mean_cl2 += CL2_vector

    mean_v /= trials
    mean_cl2 /= trials
    v_ax.plot(t, np.mean(mean_v, axis=1))
    v_ax.plot([0, t[-1]], [xi, xi], "--", c="orangered")
    v_ax.plot([0, t[-1]], [-xi, -xi], "--", c="orangered")
    v_ax.plot([0, t[-1]], [0, 0], "--", c="orangered")

    cl2_ax.plot(t, mean_cl2)
    cl2_CI = 1 / (particle_count * np.sqrt(trials))
    cl2_ax.fill_between(t, exp_CL2 + cl2_CI, max(exp_CL2 - cl2_CI, 0), alpha=0.1)

    cl2_ax.plot([0, t[-1]], [exp_CL2, exp_CL2], "--")
    cl2_ax.set(xlabel="Time", ylabel="CL2", xlim=(0, t[-1]))
    cl2_ax.ticklabel_format(axis="y", style="sci", scilimits=(-0, 1), useMathText=True)
    cl2_fig.suptitle(r"Testing $\sigma = {}^2/2$".format(np.sqrt(diffusion * 2)))
    v_fig.suptitle(r"Testing $\sigma = {}^2/2$".format(np.sqrt(diffusion * 2)))
    cl2_fig.tight_layout()

    v_ax.set(xlabel="Time", ylabel="Velocity", xlim=(0, t[-1]), ylim=(2, 4))

    cl2_fig.savefig(
        "./Overnight_Experiments/repeat_cluster_formation_cl2.jpg",
        format="jpg",
        dpi=250,
    )
    v_fig.savefig(
        "./Overnight_Experiments/repeat_cluster_formation_v.jpg", format="jpg", dpi=250
    )
    # plt.show()()


def vary_noise_in_Fig3():
    print("Running Exp 1ii)...")
    print("Plotting Figure 3...")
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 6
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    length = 10
    fig, ax = plt.subplots(3, 2, figsize=(18.0, 12.0))

    # Figure 3(a-b)
    diffusion = (0.75 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Garnier",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    GF.plot_avg_vel_CL2(ax[0, 0], ax[0, 1], t, x, v, xi, ymax=0.05)

    # Figure 3(c-d)
    diffusion = (1.0 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Garnier",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    GF.plot_avg_vel_CL2(ax[1, 0], ax[1, 1], t, x, v, xi, ymax=8e-4)
    # Figure 3(e-f)
    diffusion = (1.25 ** 2) / 2
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Garnier",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )
    # Plot average velocity and expected
    GF.plot_avg_vel_CL2(ax[2, 0], ax[2, 1], t, x, v, xi, ymax=6e-4)
    fig.suptitle("Garnier Fig 3, Vary Diffusion")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig("./Overnight_Experiments/tighterFigure3.jpg", format="jpg", dpi=250)
    # plt.show()()


def cleaner_ani_fig3():
    print("Running Exp 2i)...")
    particle_count = 2000
    diffusion = (0.5 ** 2) / 2
    well_depth = 6
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.05
    T_final = 100
    length = 10

    interaction_function = "Garnier"
    herding_function = "Garnier"

    # Set initial data for Gaussian
    mu_init = xi
    sd_init = np.sqrt(diffusion)

    gaussian = {
        "particle": normal(loc=mu_init, scale=sd_init, size=particle_count),
        "pde": lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init),
    }

    initial_data_x = None
    initial_data_v = gaussian["particle"]  # Choose indicator or gaussian
    startTime = datetime.now()
    t, x, v = run_full_particle_system(
        interaction_function=interaction_function,
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        herding_function=herding_function,
        L=length,
        well_depth=well_depth,
    )
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    plt_time = datetime.now()
    annie = hetplt.anim_full(
        t, x[:, :100], v[:, :100], L=length, mu=xi, variance=diffusion, framestep=1
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = "./Overnight_Experiments/fig3skewed"
    annie.save(fn + ".mp4", writer="ffmpeg", fps=10)
    print("Total time was {} seconds".format(datetime.now() - startTime))
    # plt.show()()


def few_particle_dist():
    print("Running Exp 3...")
    print("Plotting Figure 6...")
    dt = 0.1
    particle_count = 100
    T_final = 5 * 10 ** 4
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    diffusion = (1 ** 2) / 2
    length = 10
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 5.0))

    # Figure 6(a-b)
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion), size=particle_count),
        D=diffusion,
        interaction_function="Garnier",
        herding_function="Garnier",
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    ax[0].hist(v.flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True)
    ax[0].plot(
        np.arange(-v.max(), v.max(), 0.01),
        stats.norm.pdf(
            np.arange(-v.max(), v.max(), 0.01), loc=xi, scale=np.sqrt(diffusion)
        ),
        "--",
    )
    ax[0].set(xlabel="Velocity")

    ax[1].hist(x.flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True)
    ax[1].plot([x.min(), x.max()], [1 / (length), 1 / (length)], "--")
    fig.suptitle(
        r"Fewer Particles, Long Run $\sigma = {}^2/2, h={}, T={}$, {} particles".format(
            np.sqrt(diffusion * 2), well_depth, T_final, particle_count
        )
    )
    fig.savefig(
        "./Overnight_Experiments/few_particle_Figure6.jpg", format="jpg", dpi=250
    )
    # plt.show()()


def periodic_uniform():
    print("Running Exp 4...")
    # Which one to test? One that looks most periodic or least?
    # Both ends obv: GF7 and GF3a

    # IMPLEMENT LATER, I think the animation shows this is the case and will suffice.
    def time_average_hist(t_max):
        return


def small_well_repeat(trials=10):
    print("Running Exp 5...")
    bins = np.arange(-5, 5, 0.1)
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 5.0))

    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 4.05
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    diffusion = (1 ** 2) / 2
    length = 10
    total = np.array([])
    for _ in range(trials):
        print("Plotting Figure 6 Part {}...".format(_ + 1))
        # Figure 6(a-b)
        t, x, v = run_full_particle_system(
            particles=particle_count,
            dt=dt,
            initial_dist_v=normal(
                loc=xi, scale=np.sqrt(diffusion), size=particle_count
            ),
            D=diffusion,
            interaction_function="Garnier",
            herding_function="Garnier",
            T_end=T_final,
            L=length,
            well_depth=well_depth,
        )
        ax[0].hist(v.flatten(), bins=bins, density=True, alpha=0.1)
        total = np.concatenate((total, v.flatten()))

    ax[0].plot(
        np.arange(-v.max(), v.max(), 0.01),
        stats.norm.pdf(
            np.arange(-v.max(), v.max(), 0.01), loc=xi, scale=np.sqrt(diffusion)
        ),
        "--",
    )
    ax[0].set(xlabel="Velocity")
    ax[0].hist(total, bins=bins, density=True)
    ax[1].hist(x.flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True)
    ax[1].plot([x.min(), x.max()], [1 / (length), 1 / (length)], "--")
    fig.suptitle(
        r"Small Well $\sigma = {}^2/2, h={}, T={}$, {} particles".format(
            np.sqrt(diffusion * 2), well_depth, T_final, particle_count
        )
    )
    fig.savefig(
        "./Overnight_Experiments/small_well_repeat_mean.jpg", format="jpg", dpi=250
    )
    # plt.show()()


def breaking_clusters():
    print("Running Exp 6i)...")

    dt = 0.1
    particle_count = 200
    T_final = 1000
    exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
    well_depth = 6
    xi = 1  # 5*np.sqrt((well_depth-4)/well_depth)
    diffusion = (0.1 ** 2) / 2
    length = 10
    interaction = "Indicator"  # Short range
    # If using Garnier, you'll get bad results (like LMC)
    herding = "Smooth"
    # Gamma or cauchy initial v
    initial_data_v = normal(loc=1, scale=np.sqrt(diffusion), size=particle_count)
    left = uniform(low=0, high=0.0005, size=particle_count // 2)
    right = uniform(
        low=(length / 2), high=(length / 2) + 0.0005, size=particle_count // 2
    )
    two_clusters = np.concatenate((left, right))
    initial_data_x = two_clusters  # uniform(low=0,high=0.0005, size=particle_count)
    print(
        "\n\nSimulating {} particles with:\n {} interaction,\n {} herding,\n Diffusion is {:.4f}\n".format(
            particle_count, interaction, herding, diffusion
        )
    )
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=initial_data_v,
        initial_dist_x=initial_data_x,
        D=diffusion,
        interaction_function=interaction,
        herding_function=herding,
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 6.0))

    GF.plot_avg_vel_CL2(ax[0], ax[1], t, x, v, xi, ymax=None)

    fig2, ax2 = plt.subplots(figsize=(6.0, 6.0))
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
        title="First 20s KDE: {} particles, {} interaction,\n {} herding, Diffusion is {:.4f}".format(
            particle_count, interaction, herding, diffusion
        ),
    )

    fig4, ax4 = plt.subplots(figsize=(6.0, 6.0))
    ax4 = sns.kdeplot(
        np.repeat(t[int(500 // dt) : int(520 // dt)], particle_count),
        x[int(500 // dt) : int(520 // dt),].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax4.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(500, 520),
        ylim=(0, length),
        title="Middle 20s KDE: {} particles, {} interaction,\n {} herding, Diffusion is {:.4f}".format(
            particle_count, interaction, herding, diffusion
        ),
    )

    fn = "./Overnight_Experiments/compact_phi_xIC_point_quick"
    fig.savefig(fn + ".jpg", format="jpg", dpi=250)
    fig2.savefig(fn + "kde.jpg", format="jpg", dpi=250)
    if T_final > 20:
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
        fig3.savefig(fn + "kde.jpg", format="jpg", dpi=250)

    plt_time = datetime.now()
    annie = hetplt.anim_full(
        t,
        x[:, : particle_count // 10],
        v[:, : particle_count // 10],
        L=length,
        mu=xi,
        variance=diffusion,
        framestep=1,
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    # annie.save(fn+'.mp4',writer='ffmpeg',fps=10)
    plt.show()

    print("Running Exp 6ii)...")
    # diffusion = (1**2)/2

    initial_data_x = uniform(low=0, high=0.05, size=particle_count)
    print(
        "\n \nSimulating {} particles with:\n {} interaction,\n {} herding,\n Diffusion is {}\n".format(
            particle_count, interaction, herding, diffusion
        )
    )
    t, x, v = run_full_particle_system(
        particles=particle_count,
        dt=dt,
        initial_dist_v=initial_data_v,
        initial_dist_x=initial_data_x,
        D=diffusion,
        interaction_function=interaction,
        herding_function=herding,
        T_end=T_final,
        L=length,
        well_depth=well_depth,
    )

    fig, ax = plt.subplots(1, 2, figsize=(12.0, 5.0))

    GF.plot_avg_vel_CL2(ax[0], ax[1], t, x, v, xi, ymax=None)
    cl2_CI = 1 / (particle_count)
    ax.fill_between(t, exp_CL2 + cl2_CI, max(exp_CL2 - cl2_CI, 0), alpha=0.1)

    fig2, ax2 = plt.subplots()
    ax2 = sns.kdeplot(
        np.repeat(t[: int(20 // dt)], particle_count),
        x[: int(20 // dt),].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax2.set(xlabel="Time", ylabel="Position", xlim=(0, 20), ylim=(0, length))
    ax2.set(
        title="First 20s KDE: {} particles {} interaction,\n {} herding, Diffusion is {:.4f}".format(
            particle_count, interaction, herding, diffusion
        )
    )

    fn = "./Overnight_Experiments/compact_phi_xIC_uniform_quick"
    fig.savefig(fn + ".jpg", format="jpg", dpi=250)
    fig2.savefig(fn + "kde.jpg", format="jpg", dpi=250)
    if T_final > 20:
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
        fig3.savefig(fn + "kde.jpg", format="jpg", dpi=250)

    plt_time = datetime.now()
    annie = hetplt.anim_full(
        t,
        x[:, : particle_count // 10],
        v[:, : particle_count // 10],
        L=length,
        mu=xi,
        variance=diffusion,
        framestep=1,
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    # annie.save(fn+'.mp4',writer='ffmpeg',fps=10)
    plt.show()


if __name__ == "__main__":
    breaking_clusters()
    # repeat_cluster_formation(trials=10) #more trials and longer overnight
    # vary_noise_in_Fig3()
    #
    # small_well_repeat(trials=10) #particle count
    #
    #
    #
    # cleaner_ani_fig3()
    # few_particle_dist() #Even Longerrrrr 10 particles overnight
