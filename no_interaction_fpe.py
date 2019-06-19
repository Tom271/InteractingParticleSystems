import numpy as np
from numpy.random import normal, uniform

from scipy import sparse
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pandas import * #For visualising arrays easily
import warnings
#######################################
# PDE Approximation

def make_operators(domain_size, space_mesh):
    M, N = domain_size, space_mesh
    dv = 2*M / (N-1)
    assert dv>=0 and dv <= 0.5, "Refine the mesh"
    v = np.arange(-M, M+dv/2, dv)

    _v = np.append(-v[1:-1], [0])
    advection =  (1/dv) * (np.diag(np.append([0], _v)) - np.diag(np.array(_v), k=1) )
    diffusion = (1/(dv**2)) * (-2*np.eye(N) + np.diag(np.ones(N-1), k = -1) + np.diag(np.ones(N-1), k = 1))
    diffusion[0, 0] = -1/(dv**2)
    diffusion[-1, -1] = -1/(dv**2)

    adv = sparse.csr_matrix(advection)

    diff = sparse.csr_matrix(diffusion)

    return v, adv,  diff


def solve_advDif(D, dt, T_end, adv, diff, v, cont_init_dist):
    t=np.arange(0, T_end, dt)
    N=len(v)

    h = np.zeros((N, len(t)))
    dv= v[1] - v[0]
    if (D * dt)/(dv**2) >= 1/2:
        warnings.warn('Method may be unstable, refine parameters\n D*dt/(dv**2) = {}'.format((D * dt)/(dv**2)))
    # Setting Initial condition
    h[:, 0] = cont_init_dist(v)
    mass = []

    for j in range(1, len(t)):
        A = np.eye(N) - 0.5 * dt * (adv + D * diff)
        b = h[:,j-1] + 0.5* dt * (adv + D * diff) * h[:,j-1]
        h[:,j] = np.linalg.solve(A, b)
        if j==1 or j==len(t)-1:
            mass.append(sum(h[:,j]))


    return t, v, h, mass



#################################END of PDE ###################################

#Simulate Particle Model
def OU_process(particles = 50, iterations = 10**2, diffusion = 1, dt = 0.01, disc_init_dist = uniform(size = 50)):
    x = disc_init_dist
    traj = np.zeros((particles, iterations))
    traj[:,0] = x
    for i in range(1,iterations):
        x = x - x*dt + np.sqrt(2*diffusion*dt) * normal(size = particles)
        traj[:,i] = x
    return traj

##################### END of Particle Model #########################################


D=1
T_end = 5
dt = 0.001
domain_size = 5
space_mesh = 224

particles = 500
iterations = int(T_end/dt)

#Initial Distributions
cont_indicator = lambda x: np.array([int(i>=-1 and i<=0) for i in x])
disc_indicator = uniform(low = 4, high=0, size = particles)

cont_normal = lambda x: stats.norm.pdf(x, loc = 4, scale = 0.1)
disc_normal = normal(loc = 4, scale = 0.1, size = particles)


cont_init_dist = cont_normal
disc_init_dist = disc_normal

print('\n####### Solving PDE #########\n' + '#'*39)
print(' Diffusion Coefficient: {:s}'.format(str(D)))
print(' On interval: [-{}, {}]'.format(domain_size, domain_size))
print(' PARAMETERS:')
for p in [('Terminal time', T_end), ('Time Step', dt) , ('Space Mesh Points', space_mesh)]:
    print('  ' + '{:>22}:   {:s}'.format(*map(str,p)))
print('#'*39 + '\n')

v, adv, diff = make_operators(domain_size, space_mesh)
t, v, h, mass = solve_advDif(D, dt, T_end, adv, diff, v, cont_init_dist)


print('\n####### Solving Particle Model #########\n' + '#'*39)
print(' Diffusion Coefficient: {:s}'.format(str(D)))
print(' On interval: [-{}, {}]'.format(domain_size, domain_size))
print(' PARAMETERS:')
for p in [('Particles', particles), ('Iterations', iterations) ]:
    print('  ' + '{:>22}:   {:s}'.format(*map(str,p)))
print('#'*39 + '\n')

traj = OU_process(particles, iterations, diffusion = D, dt=dt, disc_init_dist = disc_init_dist)

#Plots
fig, ax = plt.subplots(1,1)
# ax[0].plot(traj.T)
# print('Mass loss percentage was: {:>22}% '.format( (mass[0] - mass[1])/mass[0]))
#
# ax[0].set_xlabel('Iterations')

n, bins, patches = plt.hist(traj[:,0],  bins=np.arange(-domain_size, domain_size, 0.15), density = True, label= 'OU Process')

ax.set_ylim(0,1.1)
ax.set_xlim(-domain_size, domain_size)

line, = ax.plot(v, h[:, 0], label = 'PDE')

mu = 0
sigma = np.sqrt(D)
ax.plot(v, stats.norm.pdf(v, mu, sigma), label = 'True Dist')
ax.legend()


def animate(i):
    line.set_ydata(h[:, i])  # update the data
    n, _ = np.histogram(traj[:,i],  bins=np.arange(-domain_size, domain_size, 0.15), density = True)  # update the data
    for rect, heig in zip(patches, n):
         rect.set_height(heig)



ani = animation.FuncAnimation(fig, animate, interval = 50, frames = int(T_end/dt - 1))

plt.show()
