from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import particle.plotting as hetplt
from particle.simulate import ParticleSystem

# np.random.seed(1)
sns.set()
sns.color_palette("colorblind")

parameters = {
    "interaction_function": "Gamma",
    "particles": 200,
    "D": 0,
    "initial_dist_x": "uniform_dn",
    "initial_dist_v": "neg_cauchy_dn",
    "dt": 0.001,
    "T_end": 50,
    "herding_function": "Step",
    "denominator": "Garnier",
    "gamma": 0.5,
}
startTime = datetime.now()
PS = ParticleSystem(**parameters)
t, x, v = PS.get_trajectories()

print("Time to solve was  {} seconds".format(datetime.now() - startTime))
print(t.shape, x.shape, v.shape)
print(np.mean(v[-1,]))
plt_time = datetime.now()

ani = hetplt.anim_torus(t, x, v, framestep=1, subsample=50)
print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
plt.show()
