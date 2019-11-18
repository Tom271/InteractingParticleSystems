from datetime import datetime
import numpy as np

from numpy.random import normal  # uniform
import pathlib
import pickle

# import scipy.stats as stats

import matplotlib.pyplot as plt
import seaborn as sns

from plotting import het_plot_v2 as hetplt
from particle import (
    run_full_particle_system,
    CL2,
)

sns.set()
sns.color_palette("colorblind")
file_path = "Test_Data/"
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
default_parameters = {
    "interaction_function": "Garnier",
    "particles": 2000,
    "D": 1,
    "initial_dist_x": None,
    "initial_dist_v": None,
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Garnier",
    "L": 2 * np.pi,
    "well_depth": 6,
    "denominator": "Full",
    "gamma": 1 / 10,
}

with open(file_path + "default_parameters.txt", "w") as parameter_file:
    print(str(default_parameters), file=parameter_file)

# diffusions = [0.001, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]
# diffusions = [0.5, 1]
# kwargs = dict(default_parameters)
# del kwargs["D"]
# for diffusion in diffusions:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(D=diffusion, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     plt_time = datetime.now()
#     test_data = {"Time": t, "Position": x, "Velocity": v}
#     subdir = "Diffusion/"
#     pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
#     file_name = "varyDiff_{}".format(diffusion)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))

figure_6 = {
    "interaction_function": "Gamma",
    "particles": 500,
    "D": 1 ** 2 / 2,
    "initial_dist_x": None,
    "initial_dist_v": None,
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Garnier",
    "L": 10,
    "well_depth": 10,
    "denominator": "Full",
    "gamma": 1 / 10,
}
xi = 5 * np.sqrt((figure_6["well_depth"] - 4) / figure_6["well_depth"])

figure_6["initial_dist_v"] = normal(
    loc=xi, scale=np.sqrt(figure_6["D"]), size=figure_6["particles"]
)

subdir = "Gamma/"
kwargs = dict(figure_6)
del kwargs["gamma"]
pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
with open(file_path + subdir + "parameters.txt", "w") as parameter_file:
    print(str(kwargs), file=parameter_file)
gammas = np.arange(0, 1.05, 0.05)
gammas = [0.0, 0.1, 0.5, 1.0]
for gamma in gammas:
    startTime = datetime.now()
    t, x, v = run_full_particle_system(gamma=gamma, **kwargs)
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    cl2_time = datetime.now()
    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = CL2(x[n,], L=kwargs["L"])
    test_data = {"Time": t, "Position": x, "Velocity": v, "CL2": CL2_vector}
    file_name = "{}".format(gamma)
    file_name = file_name.replace(".", "")
    pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
    print("Time to calculate cl2 was {} seconds".format(datetime.now() - cl2_time))
    print("Saved at {}\n".format(file_path + subdir + file_name))

# To load data in the future use:
with open(file_path + "default_parameters.txt", "r") as params:
    s = params.read()
    default_parameters = eval(s)
print("Using defaults:", default_parameters)

print("Reading from", file_path + subdir + file_name)
test_data = pickle.load(open(file_path + subdir + "005", "rb"))
t_1 = test_data["Time"]
x_1 = test_data["Position"]
v_1 = test_data["Velocity"]
xi = 5 * np.sqrt((figure_6["well_depth"] - 4) / figure_6["well_depth"])
length = 10
annie = hetplt.anim_full(
    t_1, x_1, v_1, mu_v=xi, variance=figure_6["D"], L=length, framestep=1
)
plt.show()
