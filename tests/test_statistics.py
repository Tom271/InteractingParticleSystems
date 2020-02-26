import numpy as np
from particle.statistics import Q_order_t, CL2  #
from particle.simulate import ParticleSystem
import matplotlib.pyplot as plt


def test_cluster_one():
    data = []
    gamma_tildes = np.arange(0, 1.01, 0.01)
    for gamma_tilde in gamma_tildes:
        pos = np.random.uniform(low=0, high=2 * np.pi, size=1000)
        data.append(Q_order_t(pos, gamma_tilde))
    print(gamma_tilde, data)
    plt.plot(gamma_tildes, 1 / np.array(data))
    plt.plot([0, 1], [1, 1], "r--")
    plt.show()
    assert Q_order_t(pos) == 1, "UhOh"


def test_one_cluster():
    PS = ParticleSystem(initial_dist_x="one_cluster")
    PS.set_velocity_initial_condition()
    PS.set_position_initial_condition()

    data = []

    gamma_tildes = np.arange(0, 1.01, 0.01)
    for gamma_tilde in gamma_tildes:
        pos = PS.x0
        data.append(Q_order_t(pos, gamma_tilde))
    print(gamma_tilde, data)
    plt.plot(gamma_tildes, 1 / np.array(data))
    plt.plot([0, 1], [1, 1], "r--")
    plt.xlabel(r"$\tilde{\gamma}$")
    plt.ylabel(r"$\frac{1}{Q_{\tilde{\gamma}}(x)}$")
    plt.show()
    # assert Q_order_t(pos) == 1, "UhOh"


def test_two_clusters():
    PS = ParticleSystem(initial_dist_x="two_clusters")
    PS.set_velocity_initial_condition()
    PS.set_position_initial_condition()
    data = []
    gamma_tildes = np.arange(0, 1.01, 0.01)
    for gamma_tilde in gamma_tildes:
        pos = PS.x0
        data.append(Q_order_t(pos, gamma_tilde))
    print(gamma_tilde, data)
    plt.plot(gamma_tildes, 1 / np.array(data))
    plt.plot([0, 1], [1, 1], "r--")
    plt.plot([0.5 - 1 / 10, 0.5 - 1 / 10], [2, 3])
    plt.plot([0.5, 0.5], [1, 2])
    plt.xlabel(r"Cluster Size as % of L $(\tilde{\gamma})$")
    plt.ylabel(r"$\frac{1}{Q_{\tilde{\gamma}}(x)}$")
    plt.show()
    # assert Q_order_t(pos) == 1, "UhOh"


def test_two_clusters_CL2():
    PS = ParticleSystem(initial_dist_x="two_clusters")
    PS.set_velocity_initial_condition()
    PS.set_position_initial_condition()
    data = []
    gamma_tildes = np.arange(0, 1.01, 0.01)
    for gamma_tilde in gamma_tildes:
        pos = PS.x0
        data.append(CL2(pos))
    plt.plot(np.array(data))
    # plt.plot([0, 1], [1, 1], "r--")
    plt.xlabel(r"CL2")
    plt.ylabel(r"CL2")
    plt.show()


Q = test_two_clusters()
test_one_cluster()
# print(Q_order_t(x))
