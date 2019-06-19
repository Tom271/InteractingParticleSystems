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
    interaction = (1/dv) * (-np.eye(N) +  np.diag(np.ones(N-1), k = 1))
    interaction[0,0] = 0
    interaction[:,-1] = 0
    diffusion = (1/(dv**2)) * (-2*np.eye(N) + np.diag(np.ones(N-1), k = -1) + np.diag(np.ones(N-1), k = 1))
    diffusion[0, 0] = -1/(dv**2)
    diffusion[-1, -1] = -1/(dv**2)

    adv = sparse.csr_matrix(advection)
    interaction = sparse.csr_matrix(interaction)
    diff = sparse.csr_matrix(diffusion)

    return v, adv, interaction, diff


def solve_advDif(D, dt, T_end, adv, interaction, diff, v, cont_init_dist):
    t=np.arange(0, T_end, dt)
    N=len(v)

    h = np.zeros((N, len(t)))
    dv= v[1] - v[0]
    if (D * dt)/(dv**2) >= 1/2:
        warnings.warn('Method may be unstable, refine parameters\n D*dt/(dv**2) = {}'.format((D * dt)/(dv**2)))
    # Setting Initial condition
    h[:, 0] = cont_init_dist(v)

    def herding_fun(u, beta):
        assert beta >= 0 , 'Beta must be greater than 0'
        herd = (u + beta * np.sign(u))/ (1 + beta)
        return herd

    def mean_in_range(v, h):
        return sum(v*h)/len(v)


    for j in range(1, len(t)):
        #M = mean_in_range(v, h[:,j-1])
        M = np.mean(h[:,j-1])
        G = herding_fun(M, 1)
        if j%(10**3) == 0:
            print('M is {}, herd fun is {}'.format(M,G))
        A = np.eye(N) - 0.5 * dt * (adv - G * interaction + D * diff)
        b = h[:,j-1] + 0.5* dt * (adv - G * interaction + D * diff) * h[:,j-1]
        h[:,j] = np.linalg.solve(A, b)

    return t, v, h



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

particles = 50
iterations = 10**4
D=1
T_end = 10
dt = 0.001
domain_size = 5
space_mesh = 224

#Initial Distributions
cont_indicator = lambda x: np.array([int(i>=-1 and i<=0) for i in x])
disc_indicator = uniform(low = -1, high=0, size = particles)

cont_normal = lambda x: stats.norm.pdf(x, loc = 1, scale = 1)
disc_normal = normal(loc = 0, scale = 1, size = particles)


cont_init_dist = cont_indicator
disc_init_dist = disc_indicator

print('\n####### Solving PDE #########\n' + '#'*39)
print(' Diffusion Coefficient: {:s}'.format(str(D)))
print(' On interval: [-{}, {}]'.format(domain_size, domain_size))
print(' PARAMETERS:')
for p in [('Terminal time', T_end), ('Time Step', dt) , ('Space Mesh Points', space_mesh)]:
    print('  ' + '{:>22}:   {:s}'.format(*map(str,p)))
print('#'*39 + '\n')

v, adv, interaction, diff = make_operators(domain_size, space_mesh)
t, v, h = solve_advDif(D, dt, T_end, adv, interaction, diff, v, cont_init_dist)


print('\n####### Solving Particle Model #########\n' + '#'*39)
print(' Diffusion Coefficient: {:s}'.format(str(D)))
print(' On interval: [-{}, {}]'.format(domain_size, domain_size))
print(' PARAMETERS:')
for p in [('Particles', particles), ('Iterations', iterations) ]:
    print('  ' + '{:>22}:   {:s}'.format(*map(str,p)))
print('#'*39 + '\n')

traj = OU_process(particles, iterations, diffusion = D, disc_init_dist = disc_normal)

#Plots
fig, ax = plt.subplots(2,1)
ax[0].plot(traj.T)

ax[0].set_xlabel('Iterations')

n, bins, patches = ax[1].hist(traj.flatten(), bins=50, density = True, label= 'OU Process')

ax[1].set_ylim(0,1.1)
ax[1].set_xlim(-domain_size, domain_size)

line, = ax[1].plot(v, h[:, 0], label = 'PDE')

mu = 1
sigma = np.sqrt(D)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax[1].plot(v, stats.norm.pdf(v, mu, sigma), label = 'True Dist')
ax[1].legend()

def init():  # only required for blitting to give a clean slate.
    line.set_ydata([np.nan] * len(v))
    return line,

def animate(i):
    line.set_ydata(h[:, i-1])  # update the data
    return line,

ani = animation.FuncAnimation(fig, animate, interval = 5, blit=True, init_func=init, save_count=50, repeat_delay = 100)

plt.show()
