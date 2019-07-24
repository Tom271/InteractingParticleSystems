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

from src import herding as herd

def run_particle_model(particles=100, D=1,
                       initial_dist_x=uniform(size=10),
                       initial_dist_v=uniform(size=10), dt=0.01, T_end=1,
                       G=herd.step_G):
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
    v[0,] = initial_dist_v
    x[0,] = initial_dist_x

    for n in range(N):
        x[n+1,] = (x[n,] + v[n,]*dt) % (2*np.pi) # Restrict to torus
        v[n+1,] = (v[n,] - v[n,]*dt + G(herd.M1_het_part(v[n,]))*dt
                     + np.sqrt(2*D*dt) * normal(size=particles))
    t = np.arange(0, T_end+dt, dt)

    return t, x, v
