from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from particle.plotting import anim_torus
from particle.simulate import ParticleSystem

# np.random.seed(1)
sns.set()
sns.color_palette("colorblind")

parameters = {
    "interaction_function": "Gamma",
    "particles": 200,
    "D": 0,
    "initial_dist_x": "uniform_dn",
    "initial_dist_v": "neg_normal_dn",
    "dt": 0.01,
    "T_end": 10,
    "herding_function": "Step",
    "denominator": "Garnier",
    "gamma": 0.5,
}
startTime = datetime.now()
PS = ParticleSystem(**parameters)
x, v = PS.get_trajectories()
t = np.arange(0, len(x) * parameters["dt"], parameters["dt"])
print("Time to solve was  {} seconds".format(datetime.now() - startTime))
print(t.shape, x.shape, v.shape)
print(np.mean(v[-1,]))
plt_time = datetime.now()

ani = anim_torus(t, x, v, framestep=1, subsample=50)
print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
plt.show()
