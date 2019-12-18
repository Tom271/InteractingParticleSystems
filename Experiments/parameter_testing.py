from datetime import datetime
import itertools

import numpy as np
import pathlib
import pickle


import seaborn as sns

from particle.simulate import run_full_particle_system, CL2

# np.random.seed(1)
sns.set()
sns.color_palette("colorblind")

file_path = "Test_Data2/"
pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)

default_parameters = {
    "interaction_function": "Garnier",
    "particles": 200,
    "D": 1,
    "initial_dist_x": "uniform_dn",
    "initial_dist_v": "gamma_dn",
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Garnier",
    "L": 2 * np.pi,
    "well_depth": 6,
    "denominator": "Full",
    "gamma": 1 / 10,
}
with open(file_path + "default_parameters.txt", "w") as parameter_file:
    print(default_parameters, file=parameter_file)

figure_6 = {
    "interaction_function": "Gamma",
    "particles": 1000,
    "D": 0.1,
    "initial_dist_x": "uniform_dn",
    "initial_dist_v": "pos_normal_dn",
    "dt": 0.1,
    "T_end": 100,
    "herding_function": "Smooth",
    "L": 10,
    "well_depth": 5,
    "denominator": "Full",
    "gamma": 1 / 10,
}


def run_and_save(parameter="D", values=[1], _filename="test", _filepath="Experiment/"):
    filename = str(_filename)
    filepath = str(_filepath)
    pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)
    kwargs = dict(default_parameters)
    with open(filepath + "parameters.txt", "w") as parameter_file:
        print(kwargs, file=parameter_file)

    print("\n Using default parameters:\n")
    for parameter_name, parameter_value in kwargs.items():
        print("\t{}:  {}".format(parameter_name, parameter_value))

    for value in values:
        print("\nSetting {} = {}".format(str(parameter), value))
        kwargs[parameter] = value
        startTime = datetime.now()
        t, x, v = run_full_particle_system(**kwargs)
        print("Time to solve was  {} seconds".format(datetime.now() - startTime))
        CL2_time = datetime.now()
        CL2_vector = np.zeros(len(t))
        for n in range(len(t)):
            CL2_vector[n] = CL2(x[n,], L=kwargs["L"])
        test_data = {
            "Time": t,
            "Position": x,
            "Velocity": v,
            "CL2": CL2_vector,
        }
        file_name = filename + "{}".format(value)
        file_name = file_name.replace(".", "")
        pickle.dump(test_data, open(filepath + file_name, "wb"))
        print(
            "Time to calculate CL2 discrepancy was  {} seconds".format(
                datetime.now() - CL2_time
            )
        )
        print("Saved at {}\n".format(filepath + file_name))


# run_and_save(
#     parameter="gamma",
#     values=np.arange(0, 1.1, 0.1),
#     _filename="GammaFullScaling",
#     _filepath=file_path + "Full/",
# )
# TODO:Use below to test all combinations?
params = {
    "denominator": ["Full", "Garnier"],
    "D": [0.05, 0.1],
    "initial_dist_x": ["uniform_dn", "one_cluster"],
    "gamma": np.arange(0, 1.1, 0.1),
}
# # Need to test Garnier herding with varying well depths (do separate to remove well depth redundancy)
keys = list(params)
begi = datetime.now()
for values in itertools.product(*map(params.get, keys)):
    # print(values)
    # filename = str(values)
    filepath1 = "".join("{}/".format(val) for val in values[:-1]).replace(".", "")
    filename = str(values[-1]).replace(".", "")
    print(filename)
    pathlib.Path(filepath1).mkdir(parents=True, exist_ok=True)
    kwargs = dict(zip(keys, values))
    with open(filepath1 + "parameters.txt", "w") as parameter_file:
        print(kwargs, file=parameter_file)

    print("\n Using default parameters:\n")
    for parameter_name, parameter_value in kwargs.items():
        print("\t{}:  {}".format(parameter_name, parameter_value))

    startTime = datetime.now()
    t, x, v = run_full_particle_system(
        **kwargs,
        dt=0.1,
        L=2 * np.pi,
        T_end=100,
        particles=1000,
        herding_function="Smooth",
        initial_dist_v="pos_normal_dn",
        interaction_function="Gamma"
    )
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    CL2_time = datetime.now()
    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = CL2(x[n,], L=2 * np.pi)
    test_data = {
        "Time": t,
        "Position": x,
        "Velocity": v,
        "CL2": CL2_vector,
    }
    pathlib.Path(file_path + filepath1).mkdir(parents=True, exist_ok=True)
    pickle.dump(test_data, open(file_path + filepath1 + filename, "wb"))
    print(
        "Time to calculate CL2 discrepancy was  {} seconds".format(
            datetime.now() - CL2_time
        )
    )
    print("Saved at {}\n".format(file_path + filepath1 + filename))
print("TOTAL TIME TAKEN:{}".format(datetime.now() - begi))
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
