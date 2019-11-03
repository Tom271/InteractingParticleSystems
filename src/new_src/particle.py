from datetime import datetime
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

# import src.herding as herd
# from src.plotting import het_plot as hetplt
#See test_sanity.py for tests


# Define herding functions
def step_G(u, beta=1):
    assert beta >= 0 , 'Beta must be greater than 0'
    return  (u + beta * np.sign(u))/ (1 + beta)

def smooth_G(u):
    return np.arctan(u)/np.arctan(1)

def no_G(u):
    return 0

def Garnier_G(u, h):
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))


# Define interaction functions
def phi_Garnier(x_i_, L=2*np.pi):
    assert L>0, "Length L must be greater than 0"
    return (L/2)*np.less_equal(x_i_, L/10, dtype=float)

def phi_indicator(x_i_):
    #TODO test for one particle.
    return  np.less_equal(x_i_, 1, dtype=float)

def phi_uniform(x_i_):
    return np.ones_like(x_i_)

def phi_zero(x_i_):
    return np.zeros_like(x_i_)

# Simulate homogeneous system
def run_hom_particle_system(particles=100,
                   D=1,
                   initial_dist=uniform(size=100),
                   dt=0.01,
                   T_end=1,
                   G=step_G,
                   well_depth=None):
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
    #TODO: check this works
    if G == Garnier_G:
        G = lambda u: Garnier_G(u, well_depth)

    t = np.arange(0, T_end + dt, dt)
    N = len(t)-1

    v = np.zeros((N+1, particles), dtype=float)

    #TODO: take density function as argument for initial data using inverse transform
    v[0,] = initial_dist

    for n in range(N):
        v[n+1,] = (v[n,] - v[n,]*dt + G(np.mean(v[n,]))*dt
                    + np.sqrt(2*D*dt) * normal(size=particles))

    return t, v

# Simulate full system

def calculate_interaction(x_curr, v_curr, phi, L):
    interaction = np.zeros(len(x_curr))
    for particle, position in enumerate(x_curr):
        distance = np.abs(x_curr - position)
        particle_interaction = phi(np.minimum(distance, L - distance))
        weighted_avg = np.sum(v_curr * particle_interaction)
        scaling = len(x_curr) #if following Garnier np.sum(particle_interaction)+10**-50 #
        interaction[particle] = weighted_avg / scaling
    return interaction

def run_full_particle_system(
    particles=100,
    D=1,
    initial_dist_x=None,
    initial_dist_v=None,
    interaction_function="Zero",
    dt=0.01,
    T_end=1,
    herding_function="Step",
    L=2*np.pi,
    well_depth=None
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

    interaction_functions = {'Garnier': lambda x: phi_Garnier(x,L),
           'Uniform': phi_uniform,
           'Zero': phi_zero,
           'Indicator': phi_indicator
          }
    try:
        phi = interaction_functions[interaction_function]
    except KeyError as error:
        print("{} is not valid. Valid interactions are {}".format(error, list(interaction_functions.keys())))
        return

    herding_functions = {"Garnier": lambda u: Garnier_G(u, well_depth),
                        "Step": lambda u: step_G(u, beta=1),
                        "Smooth": smooth_G,
                        "Zero": no_G,
                        }

    try:
        G = herding_functions[herding_function]
    except KeyError as error:
        print("{} is not valid. Valid herding functions are {}".format(error, list(herding_functions.keys())))
        return

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
        v[0,] = normal(loc=1, scale=np.sqrt(D), size=particles)
    else:
        v[0,] = initial_dist_v


    for n in range(N):
        interaction = calculate_interaction(x[n], v[n], phi, L)
        x[n + 1,] = (x[n,] + v[n,] * dt) % L  # Restrict to torus
        v[n + 1,] = (
            v[n,]
            - (v[n,] * dt)
            + G(interaction) * dt
            + np.sqrt(2*D*dt) * normal(size=particles)
        )
    t = np.arange(0, T_end + dt, dt)

    return t, x, v


def CL2(x, L=(2*np.pi)):
    '''Centered L2 discrepancy
    Adapted from https://stackoverflow.com/questions/50364048/
    python-removing-multiple-for-loops-for-faster-calculation-centered-l2-discrepa
    '''
    N  = len(x)
    term3 = 0
    term2 = np.sum(2. + np.abs(x/L - 0.5) - np.abs(x/L - 0.5)**2)
    for i in range(N):
        term3 += np.sum(1. + np.abs(x[i]/L - 0.5)/2 + np.abs(x/L - 0.5)/2 - np.abs(x[i]/L - x/L)/2)
    CL2 = (13/12) - (term2 - term3/N)/N

    return CL2


if __name__ == "__main__":

    particle_count = 500
    diffusion = 0.8
    well_depth = 6
    timestep = 0.1
    T_final = 200
    length = 2*np.pi

    interaction_function = "Garnier"
    herding_function = "Garnier"

    # Set initial data for Gaussian
    mu_init = 5*np.sqrt((well_depth-4)/well_depth)
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
    t, x, v = run_particle_model(
        interaction_function=interaction_function,
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        herding_function=herding_function,
        L=length,
        well_depth=well_depth
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


    #annie = hetplt.anim_full(t, x, v, framestep=1)
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = 'Fig4Garnier'
    #annie.save(fn+'.mp4',writer='ffmpeg',fps=10)
    print("Total time was {} seconds".format(datetime.now() - startTime))
    plt.show()
