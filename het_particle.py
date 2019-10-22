import numpy as np
from numpy.random import normal, uniform

import scipy.stats as stats
from scipy.integrate import simps

import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import herding as herd

def run_particle_model(particles=100, D=1,
                       initial_dist_x=None ,
                       initial_dist_v=None, dt=0.01, T_end=1,
                       G=herd.step_G):
    """ Space-Inhomogeneous Particle model

    Calculates the solution of the space-inhomogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist_x: Array containing initial positions of particles.
        initial_dist_v: Array containing initial velocities of particles.
        dt: Time step to be used in E-M scheme, float.
        T_end: Time point at which to end simulation, float.
        G: Interaction function - refer to herding.py.

    Returns:
        t: array of times at which velocities were calculated (only used for
           plotting).
        v: array containing velocities of each particle at every timestep.

    """
    t = np.arange(0, T_end + dt, dt)
    N = len(t)-1

    x = np.zeros((N+1, particles), dtype=float)
    v = np.zeros((N+1, particles), dtype=float)

    #TODO: take density function as argument for initial data using inverse transform
    if initial_dist_x is None:
        x[0,] = uniform(size=particles)
    else:
        x[0,] = initial_data_x

    if initial_data_v is None:
        v[0,] = uniform(size=particles)
    else:
        v[0,] = initial_data_v
    particle_list = np.array([x,v])
    for n in range(N):
        for (x_i,v_i), curr_particle in enumerate(particle_list):
                for (x_j,v_j), particle in enumerate(particle_list) if particle != curr_particle:
                    interaction_num = herd.phi_part
                    interaction_denom
        x[n+1,] = (x[n,] + v[n,]*dt) % (2*np.pi) # Restrict to torus
        v[n+1,] = (v[n,] - v[n,]*dt + G(herd.M1_het_part(v[n,]))*dt
                     + np.sqrt(2*D*dt) * normal(size=particles))
    t = np.arange(0, T_end+dt, dt)

    return t, x, v

if __name__ == "__main__":
    import herding as herd
    particle_count = 100
    diffusion = 1
    initial_data_x = np.pi/2
    timestep = 0.01
    T_final = 20
    herding_function = herd.smooth_G

    #Set initial data for Gaussian
    mu_init = -1.5
    sd_init = 0.5

    #Set max/min for indicator
    max_init = 2
    min_init = 1

    gaussian = {'particle': normal(loc=mu_init, scale=sd_init ,size=particle_count),
                'pde': lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init)}


    initial_data_v = gaussian #Choose indicator or gaussian


    t, x, v = run_particle_model(particles = particle_count, D=diffusion,
                              initial_dist_x=initial_data_x,
                              initial_dist_v=initial_data_v['particle'],
                              dt=0.01, T_end=T_final, G=herd.step_G)
    g = sns.jointplot(x.flatten(), v.flatten(), kind="hex", height=7, space=0)
    g.ax_joint.set_xlabel('Position', fontsize=16)
    g.ax_joint.set_ylabel('Velocity', fontsize=16)
    plt.show()
