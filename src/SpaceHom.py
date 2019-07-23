import numpy as np
from numpy.random import normal, uniform

import scipy.stats as stats
from scipy.integrate import simps, solve_ivp

import time
import warnings

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import herding as herd

def run_particle_model(particles=100,
                   D=1,
                   initial_dist=uniform(size=100),
                   dt=0.01,
                   T_end=1,
                   G=herd.step_G):
    """ Space-Homogeneous Particle model

    Calculates the solution of the space-homogeneous particle model using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist: Array containing initial velocities of particles.
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

    v = np.zeros((N+1, particles), dtype=float)
    M1 = np.zeros(N)
    var = np.zeros(N)

    #TODO: take density function as argument for initial data using inverse transform
    v[0,] = initial_dist

    for n in range(N):
        M1[n] = np.mean(v[n,])
        var[n] = np.var(v[n,])
        v[n+1,] = (v[n,] - v[n,]*dt + G(herd.M1_part(v[n,]))*dt
                   + np.sqrt(2*D*dt) * normal(size=particles))
    return t, v, [M1, var]


def FD_solve_hom_PDE(D=1,
              initial_dist=(lambda x: np.array([int(i>=0 and i<=1) for i in x])),
              dt=0.01, T_end=1, L=5, dv=0.1, G=herd.smooth_G):
    """ Solves the kinetic model using standard FD schemes

    Uses Crank-Nicolson and upwind techniques to approximate the solution on
    [0,T_end] given an initial condition and prints mass loss.

    Args:
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist: function describing initial density.
        dt: Time step to be used, float.
        dv: Velocity step.
        T_end: Time point at which to end simulation, float.
        L: Velocity domain of solution. Choose to be large enough so that |v|
            never exceeds L.
        G: Interaction function - refer to herding.py.
    Returns:
        v: Velocity mesh on which solution was calculated, array.
        F: Approximate solution. Array is (time, space).

    Will warn if the initial density does not approximately integrate to 1.
    """

    v = np.arange(-L, L+dv, dv)
    t = np.arange(0, T_end+dt, dt)

    J = len(v)-1
    N = len(t)-1

    F = np.zeros((N+1, J+1), dtype=float) #array with rows in time, columns in space

    F_0 = initial_dist(v) #Initial Condition
    if not np.isclose(simps(F_0, dx=dv), 1, rtol=1e-05):
        warnings.warn('Normalising initial data...')
        F_0 = F_0/simps(F_0, dx=dv)
    F[0, :] = F_0

    mu = D * dt/(dv**2)
    #TODO: Write Thomas Algorithm as a function.
    #Thomas Algorithm Coefficients
    a = 0.5*mu
    b = 1 + mu
    c = 0.5*mu

    d, e, f = [np.zeros(J+1) for _ in range(3)]


    # Build arrays of new coefficients
    M0, M1, M2 = [np.zeros(N) for _ in range(3)]

    for n in range(N):
        M0[n], M1[n], M2[n] = (herd.Mn(F[n,],v, m) for m in range(3))
        for j in range(J+1):
            if j==0 or j==J:
                continue
            herd_coeff = G(herd.M1(F[n,], v))
            inter =(dt/dv)*((v[j+1]-herd_coeff)*F[n,j+1] - (v[j]-herd_coeff)*F[n,j])
            diff = 0.5*mu * (F[n, j+1] - 2*F[n, j] + F[n, j-1])

            d[j] = F[n, j] + diff + inter
            e[j] = c/(b - a*e[j-1])
            f[j] = (d[j] + a*f[j-1]) / (b - a*e[j-1])

        for j in range(J, 0, -1):
            if j==0 or j==J:
                F[n+1,j] = 0
            else:
                F[n+1, j] = f[j] + e[j]*F[n+1, j+1]

    mass_loss =  (1 - sum(F[-1,:])/sum(F[0,:]))*100
    print('Mass loss was {:.2f}%'.format(mass_loss))
    return v, F, [M0, M1, M2]

def FV_solve_hom_PDE(D=1,
              initial_dist=(lambda x: np.array([int(i>=0 and i<=1) for i in x])),
              dt=0.01, T_end=1, L=5, dv=0.1, G=herd.smooth_G):
    """ Solves the kinetic model using a finite volume method

    Uses finite volume Euler and upwind techniques to approximate the
    solution on [0,T_end] given an initial condition and prints mass loss.

    Args:
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist: function describing initial density.
        dt: Time step to be used, float.
        dv: Velocity step.
        T_end: Time point at which to end simulation, float.
        L: Velocity domain of solution. Choose to be large enough so that |v|
            never exceeds L.
        G: Interaction function - refer to herding.py.
    Returns:
        v: Velocity mesh on which solution was calculated, array.
        F: Approximate solution. Array is (time, space).

    Will warn if the initial density does not approximately integrate to 1.
    """

    v = np.arange(-L, L+dv, dv)
    t = np.arange(0, T_end+dt, dt)

    J = len(v)-1
    N = len(t)-1

    F = np.zeros((N+1, J+1), dtype=float) #array with rows in time, columns in space

    F_0 = initial_dist(v) #Initial Condition
    if not np.isclose(simps(F_0, dx=dv), 1, rtol=1e-05):
        warnings.warn('Normalising initial data... Try increasing L')
        F_0 = F_0/simps(F_0, dx=dv)
    F[0, :] = F_0

    mu = D * dt/(dv**2)
    M0, M1, M2 = [np.zeros(N) for _ in range(3)]

    for n in range(N):
        M0[n], M1[n], M2[n] = (herd.Mn(F[n,],v, m) for m in range(3))
        for j in range(J+1):
            if j==0:
                flux_left = 0
            else:
                flux_left = flux_right

            if j < J:
                v_m = 0.5*(v[j]+v[j+1])
                herd_coeff = G(herd.M1(F[n,],v))
                adv_flux = (max(0, v_m-herd_coeff)*F[n, j+1] + min(0, v_m-herd_coeff)*F[n, j])
                diff_flux =   D*(F[n,j+1] - F[n,j])/dv
                flux_right = adv_flux + diff_flux
            else:
                flux_right = 0

            F[n+1, j] = F[n,j] + (dt/dv)*(flux_right - flux_left)

    mass_loss =  (1 - sum(F[-1,:])/sum(F[0,:]))*100
    print('Mass loss was {:.2f}%'.format(mass_loss))
    return v, F, [M0, M1, M2]

if __name__ == "__main__":
    import plotting_tools as animplt
    D = 1
    initial_dist = (lambda x: stats.norm.pdf(x, loc=2, scale=np.sqrt(2)))
    timestep = 0.001
    T_end = 100
    L = 10
    dv = 0.1

    t, v, [M1, var] = run_particle_model(particles=100,
                       D=1,
                       initial_dist=uniform(size=100),
                       dt=0.01,
                       T_end=10,
                       G=herd.step_G)

    n, bins, patches = plt.hist(v.flatten(), bins=np.arange(v.min(), v.max(), 0.15),
                               density=True, label='Velocity')
    #v, sol, moments  = FV_solve_hom_PDE(D, initial_dist, timestep, T_end, L, dv,
    #                          G=herd.smooth_G)
    #v, sol, moments  = FV_solve_hom_PDE(D, initial_dist, timestep, T_end, L, dv,
    #                          G=herd.smooth_G)
    plt.show()

    def animate_PDE(t, v, sol, step=1):
        fig, ax = plt.subplots(figsize=(20,10))
        fig.patch.set_alpha(0.0)
        ax.set_ylim(sol.min(), sol.max()+0.1)
        ax.set_xlim(v.min(), v.max())
        ax.set_xlabel('Velocity', fontsize=20)
        ax.set_ylabel('Density', fontsize=20)

        fig.suptitle('t = {}'.format(t[0]))

        mu = 1
        sigma = 1

        #v = np.arange(mu - 5*sigma, mu + 5*sigma, 0.01)
        ax.plot(v, stats.norm.pdf(v, mu, sigma), label=r'Stationary D$^{\mathrm{n}}$')

        line, = plt.plot(v, sol[0,:], label='PDE')
        ax.legend()

        def animate(i):
            line.set_ydata(sol[i*step,:])
            fig.suptitle('t = {:.2f}'.format(t[i*step]), fontsize=25)
            fig.show()

        ani = animation.FuncAnimation(fig, animate, interval=60, frames=len(t)//step)

        return ani
    # t = np.arange(0, T_end+timestep, timestep)
    # ani = animate_PDE(t, v, sol,step=50)
    # plt.show()
    # plt.plot(np.arange(0,len(moments[0])), moments[0], label='Mass')
    # plt.plot(np.arange(0,len(moments[1])), moments[1], label='Mean')
    # plt.plot(np.arange(0,len(moments[2])), moments[2]-moments[1]**2, label='Variance')
    # plt.plot([0,len(moments[0])], [1,1], 'k--')
    # plt.legend(loc='upper right')
    # plt.show()
