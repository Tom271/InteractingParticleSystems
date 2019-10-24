import numpy as np
from numpy.random import normal, uniform

import scipy.stats as stats
from scipy.integrate import simps

import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from datetime import datetime

sns.set()
sns.color_palette("colorblind")

import herding as herd
from plotting import het_plot as hetplt


def phi_Garnier(x_i_):
    return 5*np.array([float(i >= 0 and i <= (1/10)*2*np.pi) for i in x_i_])

def phi_ones(x_i_):
    return np.ones_like(x_i_)

def phi_zero(x_i_):
    return np.zeros_like(x_i_)


def G_Garnier(u, h):
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))


def run_particle_model(
    particles=100,
    D=1,
    initial_dist_x=None,
    initial_dist_v=None,
    phi=phi_zero,
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
    L = 2*np.pi
    t = np.arange(0, T_end + dt, dt)
    N = len(t) - 1

    x = np.zeros((N + 1, particles), dtype=float)
    v = np.zeros((N + 1, particles), dtype=float)

    # TODO: take density function as argument for initial data using inverse transform

    if initial_dist_x is None:
        x[0,] = uniform(low=0, high=L, size=particles)
    else:
        x[0,] = initial_dist_x

    if initial_dist_v is None:
        v[0,] = uniform(low=-2, high=2, size=particles)
    else:
        v[0,] = initial_dist_v

    interaction_vector = np.zeros(particles)
    for n in range(N):
        x_curr = x[n]
        for particle, position in enumerate(x_curr):
            dist = np.abs(x_curr - position)
            interaction = phi(np.minimum(dist, L - dist))
            numerator = np.sum(v[n, ] * interaction)
            denom = np.sum(interaction)
            interaction_vector[particle] = numerator / denom

        x[n + 1,] = (x[n,] + v[n,] * dt) % L  # Restrict to torus
        v[n + 1,] = (
            v[n,]
            - (v[n,] * dt)
            + G(interaction_vector) * dt
            + D*np.sqrt(dt) * normal(size=particles)
        )
    t = np.arange(0, T_end + dt, dt)

    return t, x, v


if __name__ == "__main__":
    import herding as herd

    particle_count = 200
    diffusion = 0.5
    well_depth = 6
    timestep = 0.1
    T_final = 100

    interaction_function = phi_Garnier
    herding_function = (lambda u: G_Garnier(u, well_depth))

    # Set initial data for Gaussian
    mu_init = 5*np.sqrt((well_depth-4)/well_depth)
    sd_init = np.sqrt(diffusion**2/ 2)

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
    t, x, v = run_particle_model(
        phi=interaction_function,
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        G=herding_function,
    )
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    # g = sns.jointplot(x.flatten(), v.flatten(), kind="hex", height=7, space=0)
    # g.ax_joint.set_xlabel("Position", fontsize=16)
    # g.ax_joint.set_ylabel("Velocity", fontsize=16)
    # plt.show()
    plt_time = datetime.now()

    fig, ax = plt.subplots()
    ax.plot(t, np.mean(v, axis=1))
    ax.set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))
    plt.savefig('avg_vel.jpg', format='jpg', dpi=1000)
    ax.plot([0, T_final],[mu_init, mu_init],'--',c='gray')
    ax.plot([0, T_final],[-mu_init, -mu_init],'--',c='gray')
    ax.plot([0, T_final],[0,0],'--',c='gray')
    fig2,ax2 = plt.subplots()
    ax2 = sns.kdeplot(np.repeat(t[:int(20//timestep)],particle_count),x[:int(20//timestep),].flatten(),shade=True, cmap=sns.cubehelix_palette(256,as_cmap=True))
    ax2.set(xlabel='Time', ylabel='Position', xlim=(0,20), ylim=(0,2*np.pi),title="First 20s KDE")

    fig2.savefig('first20kdeplot.jpg', format='jpg', dpi=1000)

    fig3,ax3 = plt.subplots()
    ax3 = sns.kdeplot(np.repeat(t[-int(20//timestep):],particle_count),x[-int(20//timestep):,].flatten(),shade=True, cmap=sns.cubehelix_palette(256,as_cmap=True))
    ax3.set(xlabel='Time', ylabel='Position',xlim=(T_final - 20, T_final),ylim=(0, 2*np.pi), title="Last 20s KDE")

    fig3.savefig('last20kdeplot.jpg', format='jpg', dpi=1000)


    annie = hetplt.anim_full(t, x, v, framestep=1)
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = 'Fig4Garnier'
    #annie.save(fn+'.mp4',writer='ffmpeg',fps=10)
    print("Total time was {} seconds".format(datetime.now() - startTime))
    plt.show()
