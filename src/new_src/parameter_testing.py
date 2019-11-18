from datetime import datetime
import numpy as np

import pathlib
import pickle

# import scipy.stats as stats

import seaborn as sns

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

ic_vs = {
    "pos_normal_dn": np.random.normal(
        loc=2, scale=np.sqrt(2), size=default_parameters["particles"]
    ),
    "neg_normal_dn": np.random.normal(
        loc=-2, scale=np.sqrt(2), size=default_parameters["particles"]
    ),
    "uniform_dn ": np.random.uniform(
        low=-5, high=5, size=default_parameters["particles"]
    ),
    "cauchy_dn": np.random.standard_cauchy(size=default_parameters["particles"]),
    "gamma_dn": np.random.gamma(
        shape=7.5, scale=1.0, size=default_parameters["particles"]
    ),
}
with open(file_path + "default_parameters.txt", "w") as parameter_file:
    print(str(default_parameters), file=parameter_file)

kwargs = dict(default_parameters)
# del kwargs["D"]
# subdir = "Diffusion/"
# pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
# for diffusion in diffusions:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(D=diffusion, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     plt_time = datetime.now()
#     test_data = {"Time": t, "Position": x, "Velocity": v}
#     file_name = "varyDiff_{:.3f}".format(diffusion)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))


figure_6 = {
    "interaction_function": "Garnier",
    "particles": 2000,
    "D": 1 ** 2 / 2,
    "initial_dist_x": None,
    "initial_dist_v": None,
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Garnier",
    "L": 10,
    "well_depth": 10,
    "denominator": "Garnier",
    "gamma": 1 / 10,
}
xi = 5 * np.sqrt((figure_6["well_depth"] - 4) / figure_6["well_depth"])
#
# figure_6["initial_dist_v"] = np.random.normal(
#     loc=xi, scale=np.sqrt(figure_6["D"]), size=figure_6["particles"]
# )
# kwargs = dict(figure_6)
# del kwargs["D"]
# diffusions = [0.001, 0.01, 0.05, 0.1, 0.5, 0.75, 1, 1.25, 1.5, 2]
#
# for diffusion in diffusions:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(D=diffusion, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     plt_time = datetime.now()
#     test_data = {"Time": t, "Position": x, "Velocity": v}
#     subdir = "Diffusion/"
#     pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
#     file_name = "varyDiff_Full_{:.3f}".format(diffusion)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))
#
# kwargs["denominator"] = "Garnier"
# for diffusion in diffusions:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(D=diffusion, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     plt_time = datetime.now()
#     test_data = {"Time": t, "Position": x, "Velocity": v}
#     subdir = "Diffusion/"
#     pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
#     file_name = "varyDiff_Garnier_{:.3f}".format(diffusion)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to pickle was  {} seconds".format(datetime.now() - plt_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))
#

# subdir = "Gamma/"
# kwargs = dict(figure_6)
# del kwargs["gamma"]
# pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
# with open(file_path + subdir + "parameters.txt", "w") as parameter_file:
#     print(str(kwargs), file=parameter_file)
# gammas = np.arange(0, 1.05, 0.05)
# gammas = [0.0, 0.1, 0.5, 1.0]
# for gamma in gammas:
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(gamma=gamma, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     cl2_time = datetime.now()
#     CL2_vector = np.zeros(len(t))
#     for n in range(len(t)):
#         CL2_vector[n] = CL2(x[n,], L=kwargs["L"])
#     test_data = {"Time": t, "Position": x, "Velocity": v, "CL2": CL2_vector}
#     file_name = "{:.3f}".format(gamma)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to calculate cl2 was {} seconds".format(datetime.now() - cl2_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))
#
#
# subdir = "Particle_Count/"
# kwargs = dict(figure_6)
# del kwargs["particles"]
# pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
# with open(file_path + subdir + "parameters.txt", "w") as parameter_file:
#     print(str(kwargs), file=parameter_file)
# particle_counts = np.arange(100, 2600, 100)
#
# for particle_count in particle_counts:
#     kwargs["initial_dist_v"] = np.random.normal(
#         loc=xi, scale=np.sqrt(kwargs["D"]), size=particle_count
#     )
#     startTime = datetime.now()
#     t, x, v = run_full_particle_system(particles=particle_count, **kwargs)
#     print("Time to solve was  {} seconds".format(datetime.now() - startTime))
#     cl2_time = datetime.now()
#     CL2_vector = np.zeros(len(t))
#     for n in range(len(t)):
#         CL2_vector[n] = CL2(x[n,], L=kwargs["L"])
#     test_data = {"Time": t, "Position": x, "Velocity": v, "CL2": CL2_vector}
#     file_name = "{}".format(particle_count)
#     file_name = file_name.replace(".", "")
#     pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
#     print("Time to calculate CL2 was {} seconds".format(datetime.now() - cl2_time))
#     print("Saved at {}\n".format(file_path + subdir + file_name))
#

kwargs = dict(figure_6)
del kwargs["initial_dist_v"]

for dn_name, ic_v in ic_vs.items():
    startTime = datetime.now()
    t, x, v = run_full_particle_system(initial_dist_v=ic_v, **kwargs)
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    cl2_time = datetime.now()
    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = CL2(x[n,], L=kwargs["L"])
    test_data = {"Time": t, "Position": x, "Velocity": v, "CL2": CL2_vector}
    subdir = "IC/"
    pathlib.Path(file_path + subdir).mkdir(parents=True, exist_ok=True)
    file_name = "{}".format(dn_name)
    pickle.dump(test_data, open(file_path + subdir + file_name, "wb"))
    print(
        "Time to calculate CL2 discrepancy was  {} seconds".format(
            datetime.now() - cl2_time
        )
    )
    print("Saved at {}\n".format(file_path + subdir + file_name))


# To load data in the future use:
# with open(file_path + "default_parameters.txt", "r") as params:
#     s = params.read()
#     default_parameters = eval(s)
# print("Using defaults:", default_parameters)
#
# print("Reading from", file_path + subdir + file_name)
# test_data = pickle.load(open(file_path + subdir + "005", "rb"))
# t_1 = test_data["Time"]
# x_1 = test_data["Position"]
# v_1 = test_data["Velocity"]
# xi = 5 * np.sqrt((figure_6["well_depth"] - 4) / figure_6["well_depth"])
# length = 10
# annie = hetplt.anim_full(
#     t_1, x_1, v_1, mu_v=xi, variance=figure_6["D"], L=length, framestep=1
# )
# plt.show()
