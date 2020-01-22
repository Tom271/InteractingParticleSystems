import numpy as np
from particle.clssimulate_gen import ParticleSystem
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.animation as animation

import particle.plotting as hetplt
from particle.statistics import avg_velocity

"""
Trying one particle cw, one ccw for oscillation of avg vel
"""
left_cluster = np.arange(np.pi - np.pi / 10, np.pi + np.pi / 10, np.pi / 250)
right_cluster = np.arange(-np.pi / 10, np.pi / 10, np.pi / 250)
default_parameters = {
    "interaction_function": "Gamma",
    "particles": 3,
    "D": 0,
    # "initial_dist_x": np.concatenate((left_cluster, right_cluster)),
    # "initial_dist_v": np.array([-1.5, 1, -1, 1, 2,2,-2,0.5,-0.5,0]),
    # "initial_dist_x": np.array(
    #     [0,1,2,3,4,5, 6 * np.pi / 100, 2 * np.pi / 3, 5 * np.pi / 3, 4 * np.pi / 3]
    # ),  # "pos_normal_dn",
    "initial_dist_x": np.array([0, 6 * np.pi / 100, 4 * np.pi / 3]),
    "initial_dist_v": np.array([1.2, 1, 0]),
    "dt": 0.001,
    "T_end": 50,
    "herding_function": "Step",
    "length": 2 * np.pi,
    "denominator": "Full",
    "gamma": 1,
}


gammas = np.arange(0.0, 0.06 + 0.005, 0.005)
tau_gamma_vec = np.zeros(len(gammas))
count = 0
avg_vel_data = []
for gamma in gammas:
    default_parameters["gamma"] = gamma
    print("Testing gamma is {}".format(default_parameters["gamma"]))
    PS = ParticleSystem(**default_parameters)
    t, x, v, tau = PS.get_trajectories(stopping_time=True)
    dt = default_parameters["dt"]
    T = default_parameters["T_end"]
    eps = 1e-03
    tau_gamma_vec[count] = t[-1]
    avg_vel_data.append(avg_velocity(v))
    count += 1

default_parameters["gamma"] = 0.5
print("Testing gamma is {}".format(default_parameters["gamma"]))
PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
dt = default_parameters["dt"]
T = default_parameters["T_end"]
eps = 1e-03
tau_1 = t[-1]

t = np.arange(0, T + dt, dt)
# plt.plot(t[: len(avg_velocity(v))], avg_velocity(v))
# plt.plot([0, tau_1], [1, 1], "g--")
# plt.plot([0, tau_1], [1 + eps, 1 + eps], "g--")
# plt.plot([0, tau_1], [1 - eps, 1 - eps], "g--")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\bar{v}$")
# plt.legend(title=r"$\gamma$", loc="right")
# plt.show()
for j in range(len(avg_vel_data)):
    with sns.color_palette("coolwarm", len(avg_vel_data)):
        plt.plot(
            t[: len(avg_vel_data[j])],
            avg_vel_data[j],
            label=r"${0:.4f}$".format(gammas[j]),
        )
#
plt.plot([0, T], [avg_velocity(v)[0], avg_velocity(v)[0]], "g--")
plt.plot([0, T], [1 + eps, 1 + eps], "g--")
plt.plot([0, T], [1 - eps, 1 - eps], "g--")
plt.plot(t[: len(avg_velocity(v))], avg_velocity(v), "k--", label="0.5")
plt.xlabel(r"$t$")
plt.ylabel(r"$\bar{v}$")
plt.legend(title=r"$\gamma$", loc="right")
plt.savefig("threepart_avg_vel.eps", transparent=True, format="eps")
plt.show()

#
plt.plot(gammas, tau_gamma_vec / tau_1)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$r_{\gamma}$")


plt.show()

default_parameters["gamma"] = 0.0
print("Testing gamma is {}".format(default_parameters["gamma"]))
PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
dt = default_parameters["dt"]
T = default_parameters["T_end"]
ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=25,
    vel_panel="line",
)
plt.show()
# writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
# ani.save("gamma01ani.mp4", writer=writer)
#
#
default_parameters["gamma"] = 0.04
print("Testing gamma is {}".format(default_parameters["gamma"]))
PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
dt = default_parameters["dt"]
T = default_parameters["T_end"]
ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=25,
    vel_panel="line",
)
writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
ani.save("3particleperiodicgamma04ani.mp4", writer=writer)

default_parameters["gamma"] = 0.06
print("Testing gamma is {}".format(default_parameters["gamma"]))
PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
dt = default_parameters["dt"]
T = default_parameters["T_end"]
ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=25,
    vel_panel="line",
)
writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
ani.save("2particleperiodicgamma06ani.mp4", writer=writer)

default_parameters["gamma"] = 0.02
print("Testing gamma is {}".format(default_parameters["gamma"]))
PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
dt = default_parameters["dt"]
T = default_parameters["T_end"]
ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=25,
    vel_panel="line",
)
writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
ani.save("2particleperiodicgamma02ani.mp4", writer=writer)

#
#
# default_parameters["gamma"] = 0.07
# print("Testing gamma is {}".format(default_parameters["gamma"]))
# PS = ParticleSystem(**default_parameters)
# t, x, v, tau = PS.get_trajectories(stopping_time=True)
# dt = default_parameters["dt"]
# T = default_parameters["T_end"]
# ani = hetplt.anim_torus(
#     t,
#     x,
#     v,
#     mu_v=1,
#     variance=1,  # np.sqrt(default_parameters["D"]),
#     framestep=25,
#     vel_panel="line",
# )
# plt.show()

# default_parameters["gamma"] = 0.03
# print("Testing gamma is {}".format(default_parameters["gamma"]))
# PS = ParticleSystem(**default_parameters)
# t, x, v, tau = PS.get_trajectories(stopping_time=True)
# dt = default_parameters["dt"]
# T = default_parameters["T_end"]
# ani = hetplt.anim_torus(
#     t,
#     x,
#     v,
#     mu_v=1,
#     variance=1,  # np.sqrt(default_parameters["D"]),
#     framestep=25,
#     vel_panel="line",
# )
# plt.show()
# writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
