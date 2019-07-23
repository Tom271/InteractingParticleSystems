import numpy as np
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.color_palette('colorblind')

from numpy.random import normal, uniform
import scipy.stats as stats

def run_OU_process(particles=100,
                   D=1,
                   initial_dist=uniform(size=100),
                   dt=0.01,
                   T_end=1):
    """ Ornstein-Uhlenbeck Process

    Calculates paths of an Ornstein-Uhlenbeck process using an
    Euler-Maruyama scheme.

    Args:
        particles: Number of particles to simulate, int.
        D: Diffusion coefficient denoted sigma in equation, float.
        initial_dist: Array containing initial velocities of particles.
        dt: Time step to be use in E-M scheme, float.
        T_end: Time point at which to end simulation, float.

    Returns:
        t: array of times at which velocities were calculated (only used for
           plotting).
        v: array containing velocities of each particle at every timestep.
        M1: array containing mean of particles' velocity at each timestep
        var: array containing variance of particles' velocity at each timestep
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
        v[n+1,] = (v[n,] - v[n,]*dt + np.sqrt(2*D*dt) * normal(size=particles))

    return t, v, [M1, var]

def FTCS(U, mu, N, J):
    """

    """
    if mu >= 0.5:
        warnings.warn('Method is likely to be unstable, mu>1/2')

    for n in range(N): #0,...,N-1
        for j in range(J+1): #0,..., J
            if j==0 or j==J:
                U[n, j] = 0 # Set BC
                continue
            U[n+1, j] = U[n,j] + mu*(U[n, j+1] - 2*U[n, j] + U[n, j-1])
    return U

def BTCS(U, mu, N, J):
    """

    """
    a = mu
    b = 1 + 2*mu
    c = mu

    d = np.zeros(J+1)
    e = np.zeros(J+1)
    f = np.zeros(J+1)

    # Build arrays of new coefficients
    for n in range(N):
        for j in range(J+1):
            if j== 0 or j==J:
                continue
            d[j] = U[n,j]
            e[j] = c/(b - a*e[j-1])
            f[j] = (d[j] + a*f[j-1]) / (b - a*e[j-1])

        for j in range(J-1, 0, -1):
            U[n+1, j] = f[j] + e[j]*U[n+1, j+1]

    return U

def CN(U, mu, N, J):
    a = 0.5 * mu
    b = 1 + mu
    c = 0.5 * mu

    d = np.zeros(J+1)
    e = np.zeros(J+1)
    f = np.zeros(J+1)

    # Build arrays of new coefficients
    for n in range(N):
        for j in range(J+1):
            if j==0 or j==J:
                continue
            d[j] = U[n, j] + 0.5* mu * (U[n, j+1] - 2*U[n, j] + U[n, j-1])
            e[j] = c/(b - a*e[j-1])
            f[j] = (d[j] + a*f[j-1]) / (b - a*e[j-1])

        for j in range(J-1, 0, -1):
            U[n+1, j] = f[j] + e[j]*U[n+1, j+1]

    return U

def solve_heat_eqn(solver=FTCS, D=1, dt=0.001, dx=0.1, T_end=5, L=5,
            initial_dist=(lambda x: stats.norm.pdf(x, loc=0, scale=1))):
    t = np.arange(0, T_end+dt, dt)
    x = np.arange(-L, L+dx , dx)

    N = len(t)-1
    J = len(x)-1
    U = np.zeros((N+1, J+1), dtype= float)

    U_0 = initial_dist(x)  #Initial Conditions
    U[0,] = U_0

    mu = D*dt/(dx**2)

    sol = solver(U, mu, N, J)

    return x, sol

def upwind(U, c, N, J):
    for n in range(N):
        for j in range(J+1):
            if j==0 or j==J:
                U[n+1,j] = 0
            else:
                U[n+1,j] = U[n,j] - (min(c,0)*(U[n,j+1]-U[n,j]) + max(c,0)*(U[n,j]-U[n, j-1]))
    return U

def FV_upwind(U, c, N, J):
    for n in range(N):
        for j in range(J+1):

            if j==0 or j==J:
                flux_left = 0
            else:
                flux_left = flux_right

            if j < J:
                flux_right = (min(c,0)*U[n, j+1] + max(c,0)*U[n, j])
            else:
                flux_right = 0

            U[n+1, j] = U[n,j] - (flux_right - flux_left)

    return U

def solve_adv_eqn(solver=upwind, a=1, dt=0.001, dx=0.1, T_end=5, L=5,
                initial_dist=(lambda x: stats.norm.pdf(x, loc=0, scale=1))):
    t = np.arange(0, T_end+dt, dt)
    x = np.arange(-L, L+dx , dx)

    N = len(t)-1
    J = len(x)-1
    U = np.zeros((N+1, J+1), dtype= float)

    U_0 = initial_dist(x) #Initial Conditions
    U[0,] = U_0
    c = a * (dt/dx)
    if abs(c)>1:
        warnings.warn('Method is likely to be unstable, CFL condition failed, c>1')

    sol = solver(U, c, N, J)
    mass_loss =  (1 - sum(U[-1,:])/sum(U[0,:]))*100
    print('{} mass loss was {:.5f}%'.format(solver.__name__, mass_loss))
    return x, sol

if __name__ == "__main__":
    colormap = plt.get_cmap('plasma')
    diff_solvers = [FTCS, BTCS, CN]
    fig, ax = plt.subplots(len(diff_solvers), 1, figsize=(10,10))
    fig.suptitle('Heat Equation in 1D', fontsize=16)
    for idx, solver in enumerate(diff_solvers):
        x, sol = solve_heat_eqn(solver=solver, dt=0.005, dx=0.1, T_end=10)
        ax[idx].set_prop_cycle(color=[colormap(k) for k in np.linspace(1, 0, 10)])
        ax[idx].set_title(solver.__name__, fontsize=14)
        ax[idx].set_xlabel(r'$x$', fontsize=16)
        ax[idx].set_ylabel(r'$u$', fontsize=16)
        for i in np.logspace(0, np.log10(len(sol[:,0])-1), num=10):
            ax[idx].plot(x, sol[int(i),])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    adv_solvers = [upwind, FV_upwind]
    fig, ax = plt.subplots(len(adv_solvers), 1, figsize=(10,10))
    fig.suptitle('Advection Equation in 1D', fontsize=16)
    for idx, solver in enumerate(adv_solvers):
        adv_x, adv_sol = solve_adv_eqn(solver=solver, a=0.5, T_end=5)
        ax[idx].set_title(solver.__name__, fontsize=14)
        ax[idx].set_xlabel(r'$x$', fontsize=14)
        ax[idx].set_ylabel(r'$u$', fontsize=14)
        ax[idx].set_prop_cycle(color=[colormap(k) for k in np.linspace(1, 0, 10)])
        for i in np.logspace(0, np.log10(np.size(adv_sol, axis=0)-1), num=10):
            ax[idx].plot(adv_x, adv_sol[int(i),])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.show()
