import numpy as np
from numpy.random import normal, uniform

import scipy.stats as stats
from scipy.integrate import simps

import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns

sns.set()
sns.color_palette("colorblind")

import herding as herd
from plotting import het_plot as hetplt


def phi(x_i_):
    out = 5 * np.array([int(i >= 0 and i <= 1) for i in x_i_])
    if len(out) == 1:
        return float(out[0])
    else:
        return out


def G_new(u, h):
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))


def run_particle_model(
    particles=100,
    D=1,
    initial_dist_x=None,
    initial_dist_v=None,
    dt=0.01,
    T_end=1,
    G=herd.step_G,
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
    t = np.arange(0, T_end + dt, dt)
    N = len(t) - 1

    x = np.zeros((N + 1, particles), dtype=float)
    v = np.zeros((N + 1, particles), dtype=float)

    # TODO: take density function as argument for initial data using inverse transform
    if initial_dist_x is None:
        x[0,] = uniform(size=particles)

    if initial_data_v is None:
        v[0,] = uniform(size=particles)

    interaction_vector = np.zeros(particles)
    for n in range(N):
        x_curr = x[n]
        for particle, position in enumerate(x_curr):
            interaction = phi(x_curr - position)
            numerator = sum(v[n, particle] * interaction) - (
                v[n, particle] * (phi([0]))
            )
            denom = sum(interaction)
            interaction_vector[particle] = numerator / denom

        x[n + 1,] = (x[n,] + v[n,] * dt) % (10)  # Restrict to torus
        v[n + 1,] = (
            v[n,]
            - v[n,] * dt
            + G(interaction_vector) * dt
            + np.sqrt(2 * D * dt) * normal(size=particles)
        )
    t = np.arange(0, T_end + dt, dt)

    return t, x, v


if __name__ == "__main__":
    import herding as herd

    particle_count = 1000
    diffusion = 1
    initial_data_x = None
    initial_data_v = normal(loc=1, scale=0.5, size=particle_count)
    timestep = 0.01
    T_final = 40
    herding_function = herd.smooth_G

    # Set initial data for Gaussian
    mu_init = -1.5
    sd_init = 0.5

    # Set max/min for indicator
    max_init = 2
    min_init = 1

    gaussian = {
        "particle": normal(loc=mu_init, scale=sd_init, size=particle_count),
        "pde": lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init),
    }

    # initial_data_v = gaussian  # Choose indicator or gaussian

    t, x, v = run_particle_model(
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=0.01,
        T_end=T_final,
        G=(lambda u: G_new(u, 5)),
    )
    # g = sns.jointplot(x.flatten(), v.flatten(), kind="hex", height=7, space=0)
    # g.ax_joint.set_xlabel("Position", fontsize=16)
    # g.ax_joint.set_ylabel("Velocity", fontsize=16)
    # plt.show()

    annie = hetplt.anim_full(t, x, v, framestep=5)
    plt.show()
