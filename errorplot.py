import numpy as np
from numpy.random import normal, uniform
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
sns.set()
sns.color_palette('colorblind')
cmap = plt.get_cmap('plasma')


from src import herding as herd
from src import ToyProblems as TP
from src import SpaceHom as hom
from src.plotting import hom_plot as homplt

savepath = './Report/Figures/'
save = False
diffusion = 1
timestep = 0.001
T_final = 10
domain_size = 6
spacestep = 0.1
particle_count = 100
herding_function = herd.smooth_G
#Set initial data for Gaussian
mu_init = -1.5
sd_init = 0.5

#Set max/min for indicator
max_init = 0
min_init = -1

gaussian = {'particle': normal(loc=mu_init, scale=sd_init ,size=particle_count),
            'pde': lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init)}

indicator = {'particle': uniform(low=min_init, high=max_init, size=particle_count),
            'pde': lambda x: np.array([int(i>=min_init and i<=max_init) for i in x])}


initial_data = gaussian #Choose indicator or Gaussian


timesteps = np.logspace(-3,-3.1,2)
print(timesteps)
error = np.zeros((2,len(timesteps)))
stat_mu =  -1
v = np.arange(-domain_size, domain_size+spacestep, spacestep)
stat_dist = stats.norm.pdf(v, stat_mu, np.sqrt(diffusion))
for idx, t in enumerate(timesteps):
    v, F_diff, moments_diff = hom.FD_solve_hom_PDE(D=diffusion,
                                      initial_dist=initial_data['pde'],
                                      dt=t, T_end=T_final, L=domain_size,
                                      dv=spacestep, G=herding_function)

    v, F_vol, moments_vol = hom.FV_solve_hom_PDE(D=diffusion,
                                      initial_dist=initial_data['pde'],
                                      dt=t, T_end=T_final, L=domain_size,
                                      dv=spacestep, G=herding_function)
    error[:,idx] = np.array([abs(stat_dist - F_vol[-1,]).sum(),
                    abs(stat_dist - F_diff[-1,]).sum()])
    print(error)

plt.plot(timesteps, error[0,])
plt.plot(timesteps, error[1,])
plt.show()
