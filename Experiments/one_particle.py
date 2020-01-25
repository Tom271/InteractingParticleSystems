import numpy as np
from particle.clssimulate_gen import ParticleSystem
import matplotlib.pyplot as plt

# import matplotlib.animation as animation

import particle.plotting as hetplt

# from particle.statistics import avg_velocity

"""
Trying one particle cw, one ccwn for oscillation of avg vel
"""
left_cluster = np.arange(np.pi - np.pi / 10, np.pi + np.pi / 10, np.pi / 250)
right_cluster = np.arange(-np.pi / 10, np.pi / 10, np.pi / 250)
default_parameters = {
    "interaction_function": "Gamma",
    "particles": 1,
    "D": 0,
    # "initial_dist_x": np.concatenate((left_cluster, right_cluster)),
    "initial_dist_x": np.array([0, 1]),  # , 6 * np.pi / 100, 4 * np.pi / 3]),
    "initial_dist_v": np.array([-(2 + 1e-11), 2]),  # , 1, 2]),  # "pos_normal_dn",
    "dt": 0.001,
    "T_end": 150,
    "herding_function": "Smooth",
    "length": 2 * np.pi,
    "denominator": "Full",
    "gamma": 0.5,
}

PS = ParticleSystem(**default_parameters)
t, x, v, tau = PS.get_trajectories(stopping_time=True)
plt.plot(t[:-1], np.mean(PS.interaction_data, axis=1), label="Mean Interaction")
plt.plot(t, np.mean(v, axis=1), label="Avg Vel")
plt.legend()
plt.show()

ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=30,
    vel_panel="line",
)
plt.show()

# gammas = np.arange(0.0, 0.1 + 0.01, 0.01)
# tau_gamma_vec = np.zeros(len(gammas))
# count = 0
# avg_vel_data = []
# for gamma in gammas:
#     default_parameters["gamma"] = gamma
#     print("Testing gamma is {}".format(default_parameters["gamma"]))
#     PS = ParticleSystem(**default_parameters)
#     t, x, v, tau = PS.get_trajectories(stopping_time=True)
#     dt = default_parameters["dt"]
#     T = default_parameters["T_end"]
#     eps = 1e-03
#     tau_gamma_vec[count] = t[-1]
#     avg_vel_data.append(avg_velocity(v))
#     count += 1
#
# default_parameters["gamma"] = 0.5
# print("Testing gamma is {}".format(default_parameters["gamma"]))
# PS = ParticleSystem(**default_parameters)
# t, x, v, tau = PS.get_trajectories(stopping_time=True)
# dt = default_parameters["dt"]
# T = default_parameters["T_end"]
# eps = 1e-03
# tau_1 = t[-1]
# plt.plot(t[1:], avg_velocity(v)[1:])
# # t = np.arange(0, T + dt, dt)
# #
# # for j in range(len(avg_vel_data)):
# #     plt.plot(
# #         t[: len(avg_vel_data[j])],
# #         avg_vel_data[j],
# #         label=r"${0:.4f}$".format(gammas[j]),
# #     )
# #
# conv_to = np.sign(avg_velocity(v)[0])
# plt.plot([0, T], [conv_to, conv_to], "g--")
# plt.plot([0, T], [conv_to + eps, conv_to + eps], "g--")
# plt.plot([0, T], [conv_to - eps, conv_to - eps], "g--")
# plt.xlabel(r"$t$")
# plt.ylabel(r"$\bar{v}$")
# # plt.legend(title=r"$\gamma$", loc="right")
# plt.show()
#
# ani = hetplt.anim_torus(t, x, v, framestep=50, vel_panel="line")
# plt.show()
# #
# #
# # plt.plot(gammas, tau_gamma_vec / tau_1)
# # plt.xlabel(r"$\gamma$")
# # plt.ylabel(r"$r_{\gamma}$")
# #
# #
# # plt.show()
# #
# # default_parameters["gamma"] = 0.03
# # print("Testing gamma is {}".format(default_parameters["gamma"]))
# # PS = ParticleSystem(**default_parameters)
# # t, x, v, tau = PS.get_trajectories(stopping_time=True)
# # dt = default_parameters["dt"]
# # T = default_parameters["T_end"]
# # ani = hetplt.anim_torus(
# #     t,
# #     x,
# #     v,
# #     mu_v=1,
# #     variance=1,  # np.sqrt(default_parameters["D"]),
# #     framestep=1,
# #     vel_panel="line",
# # )
# # plt.show()
# # # writer = animation.FFMpegWriter(fps=20, extra_args=["-vcodec", "libx264"], bitrate=2000)
# # # ani.save("gamma01ani.mp4", writer=writer)
