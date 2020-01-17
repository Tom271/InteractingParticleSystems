import numpy as np
from particle.clssimulate_gen import ParticleSystem
import matplotlib.pyplot as plt
import particle.plotting as hetplt

left_cluster = np.arange(np.pi - np.pi / 10, np.pi + np.pi / 10, np.pi / 250)
right_cluster = np.arange(-np.pi / 10, np.pi / 10, np.pi / 250)
default_parameters = {
    "interaction_function": "Gamma",
    "particles": 10,
    "D": 0,
    # "initial_dist_x": np.concatenate((left_cluster, right_cluster)),
    "initial_dist_x": np.arange(0, 2 * np.pi, 2 * np.pi / 10),
    "initial_dist_v": np.concatenate((np.zeros(9), [11])),  # "pos_normal_dn",
    "dt": 0.01,
    "T_end": 50,
    "herding_function": "Smooth",
    "length": 2 * np.pi,
    "denominator": "Full",
    "gamma": 1.0,
}
# PS = ParticleSystem(**default_parameters)
# t, x, v = PS.get_trajectories()

gammas = np.arange(0.0, 0.5 + 0.05, 0.05)
tau_gamma_vec = np.zeros(len(gammas))
count = 0
for gamma in gammas:
    default_parameters["gamma"] = gamma
    print("Testing gamma is {}".format(default_parameters["gamma"]))
    PS = ParticleSystem(**default_parameters)
    t, x, v, tau = PS.get_trajectories(stopping_time=True)
    ani = hetplt.anim_torus(
        t,
        x,
        v,
        mu_v=1,
        variance=1,  # np.sqrt(default_parameters["D"]),
        framestep=1,
        vel_panel="line",
    )
    plt.show()
    dt = default_parameters["dt"]
    T = default_parameters["T_end"]
    avg_vel = np.mean(v, axis=1)
    eps = 1e-03
    # print(np.isclose(avg_vel, 1, atol=eps))
    # print(np.where(np.isclose(avg_vel, 1,atol=eps)))
    # print(np.where(np.isclose(avg_vel, 1, atol=eps))[0][0])
    try:
        tau_gamma_vec[count] = np.where(np.isclose(avg_vel, 1, atol=eps))[0][0] * dt
    except IndexError:
        tau_gamma_vec[count] = T
    print(tau_gamma_vec[count])
    count += 1

plt.plot(gammas, tau_gamma_vec)
plt.xlabel(r"$\gamma$")
plt.ylabel(r"$r_{\gamma}$")
plt.show()
plt.plot(t, avg_vel)
plt.plot([0, T], [1, 1], "g--")
plt.plot([0, T], [1 + eps, 1 + eps], "g--")
plt.plot([0, T], [1 - eps, 1 - eps], "g--")
plt.show()

# print(avg_vel)
# converged = v[:,np.isclose(v, np.mean(v, axis=1))]

# v_bar_gamma = (len(v) - len(converged)) * dt / T
# print(v_bar_gamma)
ani = hetplt.anim_torus(
    t,
    x,
    v,
    mu_v=1,
    variance=1,  # np.sqrt(default_parameters["D"]),
    framestep=1,
    vel_panel="line",
)
plt.show()
