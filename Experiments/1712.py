from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pickle


# import seaborn as sns

from particle.simulate import run_full_particle_system, CL2
import particle.plotting as hetplt

default_parameters = {
    "interaction_function": "Gamma",
    "particles": 1000,
    "D": 0,
    "initial_dist_x": "two_clusters",
    "initial_dist_v": np.concatenate(
        (-1 * np.ones(250), 1 * np.ones(750))
    ),  # "pos_normal_dn",
    "dt": 0.1,
    "T_end": 50,
    "herding_function": "Smooth",
    "L": 2 * np.pi,
    "denominator": "Full",
    "gamma": 0,
}

filepath = "1712Data/"
filename = "ZeroNoiseDetIC_v_take2"
pathlib.Path(filepath).mkdir(parents=True, exist_ok=True)

with open(filepath + "default_parameters.txt", "w") as parameter_file:
    print(default_parameters, file=parameter_file)

startTime = datetime.now()
t, x, v = run_full_particle_system(**default_parameters)
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

pickle.dump(test_data, open(filepath + filename, "wb"))

print(
    "Time to calculate CL2 discrepancy was  {} seconds".format(
        datetime.now() - CL2_time
    )
)
print("Saved at {}\n".format(filepath + filename))
test_data = pickle.load(open(filepath + filename, "rb"))
t = test_data["Time"]
x = test_data["Position"]
v = test_data["Velocity"]

xi = 1
length = 2 * np.pi
fig1, [avg_ax, cl2_ax] = plt.subplots(1, 2, figsize=(12.0, 12.0))
# Plot average velocity and expected
particle_count = len(x[0,])
exp_CL2 = (1 / particle_count) * (5 / 4 - 13 / 12)
avg_ax.plot(t, np.mean(v, axis=1))
avg_ax.plot([0, t[-1]], [xi, xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [-xi, -xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [0, 0], "--", c="orangered")
avg_ax.set(xlabel="Time", ylabel="Average Velocity", xlim=(0, t[-1]), ylim=(-4, 4))
cl2_ax.plot(t, test_data["CL2"])
cl2_ax.plot([0, t[-1]], [exp_CL2, exp_CL2], "--")
cl2_ax.set(xlabel="Time", ylabel="CL2", xlim=(0, t[-1]))
cl2_ax.ticklabel_format(axis="y", style="sci", scilimits=(-0, 1), useMathText=True)
# plt.show()
fig1.savefig(filepath + filename + "avg.jpg", format="jpg", dpi=250)
plt.show()
# # # KDE PLOTS # # #
# fig2, ax1 = plt.subplots()
# print("...slowly ...")
# dt = default_parameters["dt"]
# ax1 = sns.kdeplot(
#     np.repeat(t[: int(20 // dt)], particle_count),
#     x[: int(20 // dt),].flatten(),
#     shade=True,
#     cmap=sns.cubehelix_palette(25, as_cmap=True),
# )
# ax1.set(
#     xlabel="Time",
#     ylabel="Position",
#     xlim=(0, 20),
#     ylim=(0, length),
#     title="First 20s KDE",
# )
# fig2.savefig(filepath + data, format="jpg", dpi=250)
# fig3, ax2 = plt.subplots()
# ax2 = sns.kdeplot(
#     np.repeat(t[-int(20 // dt) :], particle_count),
#     x[-int(20 // dt) :,].flatten(),
#     shade=True,
#     cmap=sns.cubehelix_palette(25, as_cmap=True),
#     cbar=True,
# )
# ax2.set(
#     xlabel="Time",
#     ylabel="Position",
#     xlim=(t[-1] - 20, t[-1]),
#     ylim=(0, length),
#     title="Last 20s KDE",
# )
# # plt.show()
# fig3.savefig(filepath + filename + "kde.jpg", format="jpg", dpi=250)
#
# plt.close()
# # # ANIMATION # # #
xi = 1
length = 2 * np.pi
annie = hetplt.anim_full(
    t,
    x,
    v,
    mu_v=xi,
    variance=1,  # np.sqrt(default_parameters["D"]),
    L=length,
    framestep=1,
)
# plt.show()
# if input("Save animation?"):
# writer = animation.FFMpegWriter(
#     fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000
# )
# annie.save(filepath + data + "ani.mp4", writer=writer)
plt.show()
