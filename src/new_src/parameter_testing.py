from datetime import datetime
import numpy as np

# from numpy.random import normal, uniform
import pathlib
import pickle

# import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from plotting import het_plot_v2 as hetplt
from particle import (
    run_full_particle_system,
    # CL2,
)

sns.set()
sns.color_palette("colorblind")
file_path = "Test_Data/"
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
default_parameters = {
    "interaction_function": "Garnier",
    "particles": 100,
    "D": 1,
    "initial_dist_x": None,
    "initial_dist_v": None,
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Garnier",
    "L": 2 * np.pi,
    "well_depth": 6,
    "denominator": "Full",
}

with open(file_path + "default_parameters.txt", "w") as parameter_file:
    print(str(default_parameters), file=parameter_file)

diffusions = [0.001, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]
diffusions = [0.5, 1]
kwargs = dict(default_parameters)
del kwargs["D"]
for diffusion in diffusions:
    startTime = datetime.now()
    t, x, v = run_full_particle_system(D=diffusion, **kwargs)
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    plt_time = datetime.now()
    test_data = {"Time": t, "Position": x, "Velocity": v}
    subdir = "Diffusion/"
    pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
    file_name = "varyDiff_{}".format(diffusion)
    file_name = file_name.replace(".", "")
    pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
    print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
    print("Saved at {}\n".format(file_path + subdir + file_name))

kwargs = dict(default_parameters)
del kwargs["denominator"]
scalings = ["Full", "Garnier"]
# for scaling in scalings:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(denominator=scaling, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     plt_time = datetime.now()
#     test_data = {"Time": t, "Position": x, "Velocity": v}
#     subdir = "Denominator/"
#     pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
#     file_name = "{}".format(scaling)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))

# To load data in the future use:
with open(file_path + "default_parameters.txt", "r") as params:
    s = params.read()
    default_parameters_1 = eval(s)
print(default_parameters_1)

print(file_path + subdir + file_name)
test_data = pickle.load(open(file_path + subdir + file_name, "rb"))
t_3 = test_data["Time"]
x_3 = test_data["Position"]
v_3 = test_data["Velocity"]
xi = 5 * np.sqrt(
    (default_parameters["well_depth"] - 4) / default_parameters["well_depth"]
)
length = 2 * np.pi
annie = hetplt.anim_full(
    t_3, x_3, v_3, mu_v=xi, variance=diffusion, L=length, framestep=1
)
plt.show()
