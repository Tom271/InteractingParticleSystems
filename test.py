# import matplotlib.pyplot as plt
# from matplotlib import animation
# import numpy as np
# from numpy.random import normal, uniform
# import scipy.stats as stats
#
# def herding_fun(u, beta):
#     assert beta >= 0 , 'Beta must be greater than 0'
#     herd = (u + beta * np.sign(u))/ (1 + beta)
#     return herd
#
# def mean_in_range()
# x = np.linspace(-2,2)
#
# plt.plot(x, herding_fun(x, 1))
# plt.show()

import numpy as np
from numpy.random import normal, uniform

from scipy import sparse
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

particles = 5000
iterations = 10**4
D=1
T_end = 5
dt = 0.001
domain_size = 5
space_mesh = 224

M, N = domain_size, space_mesh
dv = 2*M / (N-1)
v = np.arange(-M, M+dv/2, dv)

#Initial Distributions

disc_indicator = uniform(low = 4, high=0, size = particles)

disc_normal = normal(loc = 4, scale = 0.1, size = particles)

disc_init_dist = disc_normal


traj = OU_process(particles, iterations, diffusion = D, disc_init_dist = disc_init_dist)

#Plots
fig, ax = plt.subplots(1,1)
n, bins, patches = plt.hist(traj[:,-1],  bins=np.arange(-4,4,0.15), density = True)


ax.set_ylim(0,1.1)
ax.set_xlim(-domain_size, domain_size)

mu = 0
sigma = np.sqrt(D)
#x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
ax.plot(v, stats.norm.pdf(v, mu, sigma), label = 'True Dist')
ax.legend()


def animate(i):
    n, _ = np.histogram(traj[:,i],  bins=np.arange(-4,4,0.15), density = True)  # update the data
    for rect, h in zip(patches, n):
        rect.set_height(h)
    return patches


ani = animation.FuncAnimation(fig, animate, blit=True, interval = 50, frames = int((T_end/dt - 1)), repeat=False)

plt.show()
