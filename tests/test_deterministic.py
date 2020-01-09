from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pathlib
import pickle

from particle.simulate import run_full_particle_system
import particle.plotting as hetplt

default_parameters = {
    "interaction_function": "Gamma",
    "particles": 10,
    "D": 0.0,
    "initial_dist_x": "one_cluster",
    "initial_dist_v": np.concatenate(
        (-1 * np.ones(3), 1 * np.ones(7))
    ),  # "pos_normal_dn",
    "dt": 0.01,
    "T_end": 10,
    "herding_function": "Smooth",
    "L": 2 * np.pi,
    "denominator": "Full",
    "gamma": 1,
}

# Setting save location
save = True
filepath = "DeterministicData/"
filename = "gamma_one"
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

with open(filepath + "default_parameters.txt", "w") as parameter_file:
    print(default_parameters, file=parameter_file)

# Runnning model
startTime = datetime.now()
t, x, v = run_full_particle_system(**default_parameters)
print("Time to solve was  {} seconds".format(datetime.now() - startTime))

test_data = {
    "Time": t,
    "Position": x,
    "Velocity": v,
}

# Storing data
pickle.dump(test_data, open(filepath + filename, "wb"))

print("Saved at {}\n".format(filepath + filename))

# Reloading data (unnecessary here but useful for future)
test_data = pickle.load(open(filepath + filename, "rb"))
t = test_data["Time"]
x = test_data["Position"]
v = test_data["Velocity"]

# # # ANIMATION # # #
xi = 1
length = 2 * np.pi
fig1, avg_ax = plt.subplots(1, 1, figsize=(12.0, 12.0))

# Plot average velocity and expected
particle_count = len(x[0,])
avg_ax.plot(t, np.mean(v, axis=1))
avg_ax.plot([0, t[-1]], [xi, xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [-xi, -xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [0, 0], "--", c="orangered")
avg_ax.set(xlabel="Time", ylabel="Average Velocity", xlim=(0, t[-1]), ylim=(-4, 4))

# plt.show()
fig1.savefig(filepath + filename + "avg.jpg", format="jpg", dpi=250)

annie = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=xi,
    variance=max(0.001, np.sqrt(default_parameters["D"])),
    L=length,
    framestep=1,
)

plt.show()
if save:
    print("Saving...")
    writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"])
    annie.save(filepath + filename + "ani.mp4", writer=writer)
