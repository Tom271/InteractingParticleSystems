import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import particle.plotting as hetplt
import seaborn as sns

sns.set()
sns.color_palette("colorblind")
# # # READING DATA FROM FILE # # #
file_path = "Test_Data/"
subdir = "GarnierDenom/"

print("Reading from", file_path + subdir)

mypath = file_path + subdir
with open(mypath + "parameters.txt", "r") as params:
    s = params.read()
    default_parameters = eval(s)
print("Using defaults:\n")
for parameter_name, parameter_value in default_parameters.items():
    print("\t{}:  {}".format(parameter_name, parameter_value))

onlyfiles = [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
onlyfiles = [file for file in onlyfiles if ".txt" not in file]
print("\nData Files:\n")
print(*onlyfiles, sep="\n")


for data in onlyfiles:
    test_data = pickle.load(open(file_path + subdir + data, "rb"))
    t = test_data["Time"]
    x = test_data["Position"]
    v = test_data["Velocity"]
    if default_parameters["herding_function"] == "Garnier":
        xi = 5 * np.sqrt(
            (default_parameters["well_depth"] - 4) / default_parameters["well_depth"]
        )
        if x[0,].mean() < 0:
            xi = -xi
    else:
        xi = np.sign(x[0,].mean())
    length = default_parameters["L"]

    # # # CL2 # # #
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
    fig1.savefig(mypath + data + "avg.jpg", format="jpg", dpi=250)
    plt.close()
    # # # KDE PLOTS # # #
    # fig2, ax1 = plt.subplots()
    print("...slowly ...")
    dt = default_parameters["dt"]
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
    # fig2.savefig(mypath + data, format="jpg", dpi=250)
    fig3, ax2 = plt.subplots()
    ax2 = sns.kdeplot(
        np.repeat(t[-int(20 // dt) :], particle_count),
        x[-int(20 // dt) :,].flatten(),
        shade=True,
        cmap=sns.cubehelix_palette(25, as_cmap=True),
    )
    ax2.set(
        xlabel="Time",
        ylabel="Position",
        xlim=(t[-1] - 20, t[-1]),
        ylim=(0, length),
        title="Last 20s KDE",
    )
    # plt.show()
    fig3.savefig(mypath + data + "kde.jpg", format="jpg", dpi=250)
    plt.close()
    # # # ANIMATION # # #
    annie = hetplt.anim_full(
        t,
        x,
        v,
        mu_v=xi,
        variance=np.sqrt(default_parameters["D"]),
        L=length,
        framestep=1,
    )
    # plt.show()
    # if input("Save animation?"):
    writer = animation.FFMpegWriter(
        fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000
    )
    annie.save(mypath + data + "ani.mp4", writer=writer)
    plt.close()
    # fig1.suptitle str(default_parameters[subdir.split().lower()]))
    # fig1.tight_layout()
    # fig1.subplots_adjust(top=0.85)
    # fn = str(default_parameters[""])
    # fig1.savefig(mypath + "Figure5.jpg", format="jpg", dpi=250)
