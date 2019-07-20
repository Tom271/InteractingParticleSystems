import numpy as np
from numpy.random import normal, uniform

import scipy.stats as stats
from scipy.integrate import simps, solve_ivp

from pandas import DataFrame #For visualising arrays easily
import time
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import plotting_tools as easyplt
import herding as herd

def run_pde_model(D=1,
              initial_dist=(lambda x: np.array([int(i>=0 and i<=1) for i in x])),
              dt=0.01, T_end=1, L=5, dv=0.1, G=herd.step_G):
    """ Solves the kinetic model using standard FD schemes

    Uses standard finite-difference techniques to approximate the solution on
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

    mu = 0.5 * D * dt/(dv**2)

    v = np.arange(-L, L+dv, dv)
    t = np.arange(0, T_end+dt, dt)

    J = len(v)
    N = len(t)

    F = np.zeros((N, J), dtype= float) #array with rows in time, columns in space

    F_0 = initial_dist(v) #Initial Conditions
    if simps(F_0, dx=dv) != 1:
        warnings.warn('Normalising initial data...')
        F_0 = F_0/simps(F_0, dx = dv)
    F[0,:] = F_0


    F[:,0] = 0 #BCs
    F[:,-1] = 0
    #TODO: Write Thomas Algorithm as a function.
    #Thomas Algorithm Coefficients
    a = mu
    b = 1 + 2 * mu
    c = mu

    d = np.zeros(J)
    e = np.zeros(J)
    f = np.zeros(J)

    # Build arrays of new coefficients
    for n in range(1, N):
        for j in range(1, J-1):
            damp = (dt/dv) * (v[j] * F[n-1, j] - v[j-1] * F[n-1, j-1])
            inter = G(herd.phi_pde(F[n-1,], v))*(dt/dv) * (F[n-1, j] - F[n-1, j-1])
            diff = mu * (F[n-1, j+1] - 2*F[n-1, j] + F[n-1, j-1])

            d[j] = F[n-1, j] + diff - inter + damp
            e[j] = c/(b - a*e[j-1])
            f[j] = (d[j] + a*f[j-1]) / (b - a*e[j-1])

        for j in range(J-1, 0, -1):
            F[n,j-1] = f[j-1] + e[j-1]*F[n, j]

    mass_loss =  (1 - sum(F[-1,:])/sum(F[0,:]))*100
    print('Mass loss was {:.2f}%'.format(mass_loss))
    return v, F

def adapt_pde_model(D=1,
              initial_dist=(lambda x: np.array([int(i>=0 and i<=1) for i in x])),
              dt=0.01, T_end=1, L=5, dv=0.1, G=herd.step_G):
    """ Solves the kinetic model using standard FD schemes

    Uses standard finite-difference techniques to approximate the solution on
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

    mu = 0.5 * D * dt/(dv**2)

    v = np.arange(-L, L+dv, dv)
    t = np.arange(0, T_end+dt, dt)

    J = len(v)
    N = len(t)

    F = np.zeros((N, J), dtype= float) #array with rows in time, columns in space

    F_0 = initial_dist(v) #Initial Conditions
    if simps(F_0, dx=dv) != 1:
        warnings.warn('Normalising initial data...')
        F_0 = F_0/simps(F_0, dx = dv)
    F[0,:] = F_0


    F[:,0] = 0 #BCs
    F[:,-1] = 0
    #TODO: Write Thomas Algorithm as a function.
    #Thomas Algorithm Coefficients
    a = mu
    b = 1 + 2 * mu
    c = mu

    d = np.zeros(J)
    e = np.zeros(J)
    f = np.zeros(J)

    # Build arrays of new coefficients
    for n in range(1, N):
        for j in range(1, J-1):
#TODO: check signs here - should increase accuracy
            if v[j] >= 0: #Use backward upwind
                damp = (dt/dv) * (v[j] * F[n-1, j-1] - v[j-1] * F[n-1, j])
            else: #Forward Upwind
                damp = (dt/dv) * (v[j+1] * F[n-1, j+1] - v[j] * F[n-1, j])
            #Here they are the other way round as it is negative interaction in eq.
            if G(herd.phi_pde(F[n-1,], v)) <= 0:
                inter = G(herd.phi_pde(F[n-1,], v))*(dt/dv) * (F[n-1, j-1] - F[n-1, j])
            else:
                inter = G(herd.phi_pde(F[n-1,], v))*(dt/dv) * (F[n-1, j+1] - F[n-1, j])

            #damp = (dt/dv) * (v[j] * F[n-1, j] - v[j-1] * F[n-1, j-1])
            diff = mu * (F[n-1, j+1] - 2*F[n-1, j] + F[n-1, j-1])

            d[j] = F[n-1, j] + diff - inter + damp
            e[j] = c/(b - a*e[j-1])
            f[j] = (d[j] + a*f[j-1]) / (b - a*e[j-1])

        for j in range(J-1, 0, -1):
            F[n,j-1] = f[j-1] + e[j-1]*F[n, j]

    mass_loss =  (1 - sum(F[-1,:])/sum(F[0,:]))*100
    print('Mass loss was {:.2f}%'.format(mass_loss))
    return v, F

    diffusion = 1

initial_dist = (lambda x: np.array([int(i>=0 and i<=1) for i in x]))
timestep = 0.01
T_end = 10
L = 5
dv= 0.1
diffusion = 1

v, adapt_sol = adapt_pde_model(diffusion, initial_dist, timestep, T_end, L, dv, G=herd.smooth_G)
v, sol = run_pde_model(diffusion, initial_dist, timestep, T_end, L, dv, G=herd.smooth_G)

mu = 1

plt.plot(v, sol[-1,:], label = 'Static Solver')
plt.plot(v, adapt_sol[-1,:], label = 'Adapted Solver')
v = np.arange(mu - 5*diffusion, mu + 5*diffusion, 0.01)
plt.plot(v, stats.norm.pdf(v, mu, diffusion),
           label=r'$\mathcal{N}(%g,%g)$'% (mu,diffusion))
plt.legend()
plt.show()
