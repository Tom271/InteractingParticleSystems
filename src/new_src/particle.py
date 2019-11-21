from datetime import datetime
import numpy as np
from numpy.random import normal, uniform
import scipy.stats as stats

# from scipy.integrate import simps
# import warnings

import matplotlib.pyplot as plt
import seaborn as sns

# import matplotlib.animation as animation

from plotting import het_plot_v2 as hetplt

sns.set()
sns.color_palette("colorblind")

# See test_sanity.py for tests


# Define herding functions
def step_G(u, beta=1):
    assert beta >= 0, "Beta must be greater than 0"
    return (u + beta * np.sign(u)) / (1 + beta)


def smooth_G(u):
    return np.arctan(u) / np.arctan(1)


def no_G(u):
    return 0


def Garnier_G(u, h):
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))


# Define interaction functions
def phi_zero(x_i_):
    return np.zeros_like(x_i_)


def phi_uniform(x_i_):
    return np.ones_like(x_i_)


def phi_indicator(x_i_):
    # TODO test for one particle.
    return 5 * np.less_equal(x_i_, 0.01, dtype=float)


def phi_Garnier(x_i_, L=2 * np.pi):
    assert L > 0, "Length L must be greater than 0"
    return (L / 2) * np.less_equal(x_i_, L / 10, dtype=float)


def phi_gamma(x_i_, gamma=1 / 10, L=2 * np.pi):
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    assert L > 0, "Length L must be greater than 0"
    return ((L * gamma) / 2) * np.less_equal(x_i_, gamma * L, dtype=float)


def phi_smoothed_indicator(x, a):
    f = np.zeros(len(x))
    for i in range(len(x)):
        if a <= np.abs(x[i]) <= a + 1:
            f[i] = np.exp(1 / (x[i] ** 2 - (a + 1) ** 2)) / np.exp(
                1 / (a ** 2 - (a + 1) ** 2)
            )
        elif np.abs(x[i]) < a:
            f[i] = 1
        else:
            f[i] = 0
    return f


# Simulate homogeneous system
def run_hom_particle_system(
    particles=100,
    D=1,
    initial_dist=uniform(size=100),
    dt=0.01,
    T_end=1,
    G=step_G,
    well_depth=None,
):
    """ Space-Homogeneous Particle model

    Calculates the solution of the space-homogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist: Array containing initial velocities of particles.
        dt: Time step to be use in E-M scheme, float.
        T_end: Time point at which to end simulation, float.
        G: Interaction function chosen from dictionary.

    Returns:
        t: array of times at which velocities were calculated (only used for
           plotting).
        v: array containing velocities of each particle at every timestep.

    """
    if G == Garnier_G:

        def G(u):
            return Garnier_G(u, well_depth)

    t = np.arange(0, T_end + dt, dt)
    N = len(t) - 1

    v = np.zeros((N + 1, particles), dtype=float)

    # TODO: take density function as argument for initial data using inverse transform
    v[0,] = initial_dist

    for n in range(N):
        v[n + 1,] = (
            v[n,]
            - v[n,] * dt
            + G(np.mean(v[n,])) * dt
            + np.sqrt(2 * D * dt) * normal(size=particles)
        )

    return t, v


# Simulate full system


def calculate_interaction(x_curr, v_curr, phi, L, denominator="Full"):
    interaction = np.zeros(len(x_curr))
    for particle, position in enumerate(x_curr):
        distance = np.abs(x_curr - position)
        particle_interaction = phi(np.minimum(distance, L - distance))
        weighted_avg = np.sum(v_curr * particle_interaction)
        if denominator == "Full":
            scaling = np.sum(particle_interaction) + 10 ** -50
        if denominator == "Garnier":
            scaling = len(x_curr)
        interaction[particle] = weighted_avg / scaling
    return interaction


# def calculate_interaction(x_curr, v_curr, phi, L, denominator="Full"):
#     interaction = np.zeros(len(x_curr))
#     # TODO: X - X^T for pairwise distance test against current method
#     X = np.tile(x_curr, (len(x_curr), 1))
#     V = np.tile(v_curr,(len(v_curr),1))
#     N = len(x_curr)
#     # X = np.repeat(x_curr[:,np.newaxis] , N , axis=1)
#     # V = np.repeat(v_curr[np.newaxis,:], N, axis=0)
#
#     distance = np.abs(X - X.transpose())
#     particle_interaction = phi(np.minimum(distance, L - distance))
#     weighted_avg = np.sum(V * particle_interaction, axis=1)
#     if denominator == "Full":
#         scaling = np.sum(particle_interaction, axis=1) + 10 ** -50
#     if denominator == "Garnier":
#         scaling = N
#     interaction = weighted_avg / scaling
#     return interaction


def run_full_particle_system(
    particles=100,
    D=1,
    initial_dist_x=None,
    initial_dist_v=None,
    interaction_function="Zero",
    dt=0.01,
    T_end=1,
    herding_function="Step",
    L=2 * np.pi,
    denominator="Full",
    well_depth=None,
    gamma=1 / 10,
):
    """ Space-Inhomogeneous Particle model

    Calculates the solution of the space-inhomogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist_x: Array containing initial positions of particles.
        initial_dist_v: Array containing initial velocities of particles.
        dt: Time step to be use in E-M scheme, float.
        T_end: Time point at which to end simulation, float.
        G: Interaction function - refer to herding.py

    Returns:
        t: array of times at which velocities were calculated (only used for
           plotting).
        v: array containing velocities of each particle at every timestep.

    """

    interaction_functions = {
        "Garnier": lambda x: phi_Garnier(x, L),
        "Uniform": phi_uniform,
        "Zero": phi_zero,
        "Indicator": phi_indicator,
        "Smoothed Indicator": phi_smoothed_indicator,
        "Gamma": lambda x: phi_gamma(x, gamma, L),
    }
    try:
        phi = interaction_functions[interaction_function]
    except KeyError as error:
        print(
            "{} is not valid. Valid interactions are {}".format(
                error, list(interaction_functions.keys())
            )
        )
        return

    herding_functions = {
        "Garnier": lambda u: Garnier_G(u, well_depth),
        "Step": lambda u: step_G(u, beta=1),
        "Smooth": smooth_G,
        "Zero": no_G,
    }

    try:
        G = herding_functions[herding_function]
    except KeyError as error:
        print(
            "{} is not valid. Valid herding functions are {}".format(
                error, list(herding_functions.keys())
            )
        )
        return

    t = np.arange(0, T_end + dt, dt)
    N = len(t) - 1

    x = np.zeros((N + 1, particles), dtype=float)
    v = np.zeros((N + 1, particles), dtype=float)

    # TODO: take density function as argument for initial data using inverse transform
    left_cluster = np.random.uniform(low=0, high=0.0005, size=(particles // 2))
    right_cluster = np.random.uniform(
        low=(L / 2), high=(L / 2) + 0.0005, size=(particles // 2)
    )
    ic_xs = {
        "uniform_dn": np.random.uniform(low=0, high=L, size=particles),
        "one_cluster": 0.0,
        "two_clusters": np.concatenate((left_cluster, right_cluster)),
    }
    # Hack if odd number of particles is passed
    if len(ic_xs["two_clusters"]) != particles:
        ic_xs["two_clusters"] = np.concatenate((ic_xs["two_clusters"], np.array([0.0])))
    try:
        x[0,] = ic_xs[initial_dist_x]
    except (KeyError, TypeError) as error:
        if isinstance(initial_dist_x, (list, tuple, np.ndarray)):
            print("Using ndarray")
            x[0,] = initial_dist_x
        else:
            print(
                "{} is not a valid keyword. Valid initial conditions for position are {}".format(
                    error, list(ic_xs.keys())
                )
            )
            print("Using default, uniform distrbution\n")
            x[0,] = uniform(low=0, high=L, size=particles)

    ic_vs = {
        "pos_normal_dn": np.random.normal(loc=2, scale=np.sqrt(2), size=particles),
        "neg_normal_dn": np.random.normal(loc=-2, scale=np.sqrt(2), size=particles),
        "uniform_dn": np.random.uniform(low=-5, high=5, size=particles),
        "cauchy_dn": np.random.standard_cauchy(size=particles),
        "gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=particles),
    }
    try:
        v[0,] = ic_vs[initial_dist_v]
    except (KeyError, TypeError) as error:
        if isinstance(initial_dist_v, (list, tuple, np.ndarray)):
            print("Using ndarray for velocity distribution")
            v[0,] = initial_dist_v
        else:
            print(
                "{} is not a valid keyword. Valid initial conditions for velocity are {}".format(
                    error, list(ic_vs.keys())
                )
            )
            print("Using default, positive normal distrbution\n")
            v[0,] = normal(loc=1, scale=np.sqrt(D), size=particles)

    for n in range(N):
        interaction = calculate_interaction(x[n], v[n], phi, L, denominator)
        x[n + 1,] = (x[n,] + v[n,] * dt) % L  # Restrict to torus
        v[n + 1,] = (
            v[n,]
            - (v[n,] * dt)
            + G(interaction) * dt
            + np.sqrt(2 * D * dt) * normal(size=particles)
        )
    t = np.arange(0, T_end + dt, dt)

    return t, x, v


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


if __name__ == "__main__":

    particle_count = 2000
    diffusion = (0.5 ** 2) / 2
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 100
    length = 2 * np.pi

    interaction_function = "Garnier"
    herding_function = "Garnier"

    # Set initial data for Gaussian
    mu_init = xi
    sd_init = np.sqrt(diffusion)

    # Set max/min for indicator
    max_init = 2
    min_init = 1

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
    # g = sns.jointplot(x.flatten(), v.flatten(), kind="hex", height=7, space=0)
    # g.ax_joint.set_xlabel("Position", fontsize=16)
    # g.ax_joint.set_ylabel("Velocity", fontsize=16)
    # plt.show()
    plt_time = datetime.now()
    # model_prob_x, _ = np.histogram(x[-500:-1,].flatten(), bins=np.arange(x.min(), x.max(), 0.15),
    #                                      density=True)
    # model_prob_v, _ = np.histogram(v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15),
    #                                 density=True)
    # model_prob_x, _ = np.histogram(x[-1,], bins=np.arange(x.min(), x.max(), 0.15),
    #                                      density=True)
    # model_prob_v, _ = np.histogram(v[-1,], bins=np.arange(v.min(), v.max(), 0.15),
    #                                 density=True)
    # fig, ax = plt.subplots(1,2, figsize=(24 ,12))
    # ax[0].hist(x[-1,], bins=np.arange(x.min(), x.max(), 0.15), density=True)
    # ax[0].plot([x.min(),x.max()], [1/length ,1/length], '--')
    # ax[0].set(xlabel='Position')
    #
    # ax[1].hist(v[-1,], bins=np.arange(v.min(), v.max(), 0.15),
    #                                       density=True)
    # ax[1].plot(np.arange(-v.max(),v.max(),0.01), stats.norm.pdf(np.arange(-v.max(),v.max(),0.01), loc=xi, scale=np.sqrt(diffusion)), '--')
    # ax[1].set(xlabel='Velocity')
    # true_prob_x = 1/(2*np.pi)*np.ones(len(model_prob_x))
    # true_prob_v = stats.norm.pdf(np.arange(v.min(), v.max()-0.15, 0.15), loc=0, scale=np.sqrt(diffusion))
    # fig.savefig('smallwellxvhist.jpg', format='jpg', dpi=250)

    # print("KL Divergence of velocity distribution:",     stats.entropy(model_prob_v, true_prob_v))
    # annie = hetplt.anim_full(t, x, v, mu=xi, variance=diffusion, L=length, framestep=1)
    annie = hetplt.anim_full(
        t, x, v, mu_v=xi, variance=diffusion, L=length, framestep=1
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = "indic_strong_cluster_phi_sup"
    plt.show()
    # annie.save(fn + ".mp4", writer="ffmpeg", fps=10)
    print("Total time was {} seconds".format(datetime.now() - startTime))
