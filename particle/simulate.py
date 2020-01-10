from datetime import datetime
import numpy as np
import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

import particle.interactionfunctions as phis
import particle.herdingfunctions as Gs


sns.set()
sns.color_palette("colorblind")

# See test_sanity.py for tests

# Simulate homogeneous system


def run_hom_particle_system(
    particles=100,
    D=1,
    initial_dist_v=None,
    dt=0.1,
    T_end=1,
    herding_function="Step",
    well_depth=None,
    gamma=1 / 10,
):
    """ Space-Homogeneous Particle model

    Calculates the solution of the space-homogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist_v: String corresponding to dictionary item or array containing
                        initial velocities of particles.
        dt: Time step to be use in Euler-Maruyama scheme, float.
        T_end: Time point at which to end simulation, float.
        herding_function: String corresponding to dictionary item.
        well_depth: float to be passed to the Garnier herding function.
        gamma: float to be passed to the gamma interaction function
    Returns:
        t: array of times at which velocities were calculated (only used for
           plotting).
        v: array containing velocities of each particle at every timestep.

        Typical Usage:
            t,v = run_hom_particle_system()

    See also: :py:mod:'~particle.interactionfunctions', :py:mod:'~particle.herdingfunctions'
    """

    herding_functions = {
        "Garnier": lambda u: Gs.Garnier(u, well_depth),
        "Step": lambda u: Gs.step(u, beta=1),
        "Smooth": Gs.smooth,
        "Zero": Gs.zero,
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

    v = np.zeros((N + 1, particles), dtype=float)

    # TODO: take density function as argument for initial data using inverse transform
    if initial_dist_v is None:
        print("Using default, positive normal distrbution\n")
        v[0,] = np.random.normal(loc=1, scale=np.sqrt(D), size=particles)

    for n in range(N):
        v[n + 1,] = (
            v[n,]
            - v[n,] * dt
            + G(np.mean(v[n,])) * dt
            + np.sqrt(2 * D * dt) * np.random.normal(size=particles)
        )

    return t, v


# Simulate full system


def calculate_interaction(x_curr, v_curr, phi, L, denominator="Full"):
    """Calculate interaction term of the full particle system

        Args:
            x_curr: np.array of current particle positions
            v_curr: np.array of current particle velocities
            phi: interaction function
            L: domain length, float
            denominator: string corresponding to scaling by the total number of
            particles or the number of particles that are interacting with each particle

        Returns:
            interaction_vector: vector containing the interaction at the current
             time step for each particle

        See also: :py:mod:'~particle.interactionfunctions'
    """
    interaction_vector = np.zeros(len(x_curr))
    for particle, position in enumerate(x_curr):
        distance = np.abs(x_curr - position)
        particle_interaction = phi(np.minimum(distance, L - distance))
        weighted_avg = np.sum(v_curr * particle_interaction)
        if denominator == "Full":
            scaling = np.sum(particle_interaction) + 10 ** -50
        if denominator == "Garnier":
            scaling = len(x_curr)
        interaction_vector[particle] = weighted_avg / scaling
    return interaction_vector


# def calculate_interaction(x_curr, v_curr, phi, L, denominator="Full"):
#     N = len(x_curr)
#     # TODO: X - X^T for pairwise distance test against current method
#     X = np.tile(x_curr, (N, 1))
#     V = np.tile(v_curr,(N,1))

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
    dt=0.1,
    T_end=100,
    herding_function="Step",
    L=2 * np.pi,
    denominator="Full",
    well_depth=None,
    gamma=1 / 10,
):
    """ Full Particle model

    Calculates the solution of the space-inhomogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles (int): Number of particles to simulate.
        D (float): Diffusion coefficient denoted sigma in equation.
        initial_dist_x (string, array_like): The initial positions of the particles.
        initial_dist_v (string, array_like): The initial velocities of the particles.
        dt (float): Time step to be use in E-M scheme.
        T_end (float): Time point at which to end simulation.
        herding_function (string): Choice of herding function.
        L (float): Domain length, must be positive.
        Denominator (string): Either "Full" or "Garnier", scales the interaction term
            by either the number of particles each particle is interacting with or the
            total number of particles in the system.

    Returns:
        t (array): Times at which velocities were calculated (only used for
           plotting).
        x (array): Positions of each particle at every timestep.
        v (array): Velocities of each particle at every timestep.

    Usage:
        t,x,v = run_full_particle_system()

    See also: :py:mod:`~particle.interactionfunctions`, :py:mod:`~particle.herdingfunctions`, :py:func:`calculate_interaction`
    """

    # Get interaction function from dictionary, if not valid, throw error
    interaction_functions = {
        "Garnier": lambda x: phis.Garnier(x, L),
        "Uniform": phis.uniform,
        "Zero": phis.zero,
        "Indicator": lambda x: phis.indicator(x, L),
        "Smoothed Indicator": phis.smoothed_indicator,
        "Gamma": lambda x: phis.gamma(x, gamma, L),
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
    # Get herding function from dictionary, if not valid, throw error
    herding_functions = {
        "Garnier": lambda u: Gs.Garnier(u, well_depth),
        "Step": lambda u: Gs.step(u, beta=1),
        "Smooth": Gs.smooth,
        "Zero": Gs.zero,
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

    # Set initial data in space
    # TODO: take density function as argument for initial data using inverse transform
    left_cluster = np.random.uniform(
        low=(np.pi / 2) - np.pi / 10,
        high=(np.pi / 2) + np.pi / 10,
        size=(particles // 2),
    )
    right_cluster = np.random.uniform(
        low=(3 * np.pi / 2) - np.pi / 10,
        high=(3 * np.pi / 2) + np.pi / 10,
        size=(particles // 2),
    )
    ic_xs = {
        "uniform_dn": np.random.uniform(low=0, high=L, size=particles),
        "one_cluster": np.concatenate((left_cluster, left_cluster)),
        "two_clusters": np.concatenate((left_cluster, right_cluster)),
    }
    # Hack if odd number of particles is passed
    if len(ic_xs["two_clusters"]) != particles:
        ic_xs["two_clusters"] = np.concatenate((ic_xs["two_clusters"], np.array([0.0])))
    # Try using dictionary to get IC, if not check if input is array, else use a
    # default IC
    try:
        x[0,] = ic_xs[initial_dist_x]
    except (KeyError, TypeError) as error:
        if isinstance(initial_dist_x, (list, tuple, np.ndarray)):
            print("Using ndarray")
            x[0,] = initial_dist_x
        elif initial_dist_x is None:
            print("Using default, uniform distrbution\n")
            x[0,] = np.random.uniform(low=0, high=L, size=particles)
        else:
            print(
                "{} is not a valid keyword. Valid initial conditions for position are {}".format(
                    error, list(ic_xs.keys())
                )
            )
    # Initial condition in velocity
    ic_vs = {
        "pos_normal_dn": np.random.normal(loc=1, scale=np.sqrt(2), size=particles),
        "neg_normal_dn": np.random.normal(loc=-1, scale=np.sqrt(2), size=particles),
        "uniform_dn": np.random.uniform(low=0, high=1, size=particles),
        "cauchy_dn": np.random.standard_cauchy(size=particles),
        "gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=particles),
    }
    # Try using dictionary to get IC, if not check if input is array, else use a
    # default IC
    try:
        v[0,] = ic_vs[initial_dist_v]
    except (KeyError, TypeError) as error:
        if isinstance(initial_dist_v, (list, tuple, np.ndarray)):
            print("Using ndarray for velocity distribution")
            v[0,] = initial_dist_v
        elif initial_dist_v is None:
            print("Using default, positive normal distrbution\n")
            v[0,] = np.random.normal(loc=1, scale=np.sqrt(D), size=particles)
        else:
            print(
                "{} is not a valid keyword. Valid initial conditions for velocity are {}".format(
                    error, list(ic_vs.keys())
                )
            )
    # Solving the system using an Euler-Maruyama scheme
    for n in range(N):
        interaction = calculate_interaction(x[n], v[n], phi, L, denominator)
        x[n + 1,] = (x[n,] + v[n,] * dt) % L  # Restrict to torus
        v[n + 1,] = (
            v[n,]
            - (v[n,] * dt)
            + G(interaction) * dt
            + np.sqrt(2 * D * dt) * np.random.normal(size=particles)
        )
    t = np.arange(0, T_end + dt, dt)

    return t, x, v


def CL2(x, L=(2 * np.pi)):
    """Centered L2 discrepancy

    Calculate the squared centred L2 discrepancy for quantifying uniformity.

    Args:
        x: vector containing particle positions at a given time
        L: domain length, float

    Returns:
        CL2: the CL2 discrepancy at this time, float

    Adapted from https://stackoverflow.com/questions/50364048/python-removing-multiple-for-loops-for-faster-calculation-centered-l2-discrepa
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

    particle_count = 200
    diffusion = (0.5 ** 2) / 2
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 100
    length = 2 * np.pi

    interaction_function = "Uniform"
    herding_function = "Step"

    # Set initial data for Gaussian
    mu_init = xi
    sd_init = np.sqrt(diffusion)

    # Set max/min for indicator
    max_init = 2
    min_init = 1

    gaussian = {
        "particle": np.random.normal(loc=mu_init, scale=sd_init, size=particle_count),
        "pde": lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init),
    }

    initial_data_x = None
    initial_data_v = gaussian["particle"]  # Choose indicator or gaussian
    startTime = datetime.now()
    t, v = run_hom_particle_system(
        # interaction_function=interaction_function,
        particles=particle_count,
        D=diffusion,
        # initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        herding_function=herding_function,
        # L=length,
        well_depth=well_depth,
        gamma=1 / 10,
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
    # annie = hetplt.anim_full(
    #     t, x, v, mu_v=xi, variance=diffusion, L=length, framestep=1
    # )

    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = "indic_strong_cluster_phi_sup"
    plt.show()
    # annie.save(fn + ".mp4", writer="ffmpeg", fps=10)
    print("Total time was {} seconds".format(datetime.now() - startTime))
