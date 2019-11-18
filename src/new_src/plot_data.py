import matplotlib.pyplot as plt
import numpy as np
import pickle
from plotting import het_plot_v2 as hetplt
import seaborn as sns

sns.set()
sns.color_palette("colorblind")
# # # READING DATA FROM FILE # # #
file_path = "Test_Data/"
with open(file_path + "default_parameters.txt", "r") as params:
    s = params.read()
    default_parameters = eval(s)
print("Using defaults:", default_parameters)

subdir = "Gamma/"

print("Reading from", file_path + subdir)
test_data = pickle.load(open(file_path + subdir + "1", "rb"))
t = test_data["Time"]
x = test_data["Position"]
v = test_data["Velocity"]

xi = 5 * np.sqrt((10 - 4) / 10)
length = 10
# # # ANIMATION # # #
annie = hetplt.anim_full(t, x, v, mu_v=xi, variance=0.5, L=length, framestep=1)
plt.show()

# # # CL2 # # #
fig, [avg_ax, cl2_ax] = plt.subplots(1, 2, figsize=(12.0, 12.0))
# Plot average velocity and expected
particle_count = len(x[0,])
exp_CL2 = 1 / particle_count * (5 / 4 - 13 / 12)
avg_ax.plot(t, np.mean(v, axis=1))
avg_ax.plot([0, t[-1]], [xi, xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [-xi, -xi], "--", c="orangered")
avg_ax.plot([0, t[-1]], [0, 0], "--", c="orangered")
avg_ax.set(xlabel="Time", ylabel="Average Velocity", xlim=(0, t[-1]), ylim=(-4, 4))
cl2_ax.plot(t, test_data["CL2"])
cl2_ax.plot([0, t[-1]], [exp_CL2, exp_CL2], "--")
cl2_ax.set(xlabel="Time", ylabel="CL2", xlim=(0, t[-1]))
cl2_ax.ticklabel_format(axis="y", style="sci", scilimits=(-0, 1), useMathText=True)
plt.show()

# # # KDE PLOTS # # #
# fig1, ax1 = plt.subplots()
# print("...slowly ...")
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
# fig2, ax2 = plt.subplots()
# ax2 = sns.kdeplot(
#     np.repeat(t[-int(20 // dt) :], particle_count),
#     x[-int(20 // dt) :,].flatten(),
#     shade=True,
#     cmap=sns.cubehelix_palette(25, as_cmap=True),
# )
# ax2.set(
#     xlabel="Time",
#     ylabel="Position",
#     xlim=(T_final - 20, T_final),
#     ylim=(0, length),
#     title="Last 20s KDE",
# )
