import numpy as np


# Define interaction functions
def zero(x_i):
    """No interaction between particles"""
    return np.zeros_like(x_i)


def uniform(x_i):
    """All particles interact with every other particle equally"""
    return np.ones_like(x_i)


def indicator(x_i, L=2 * np.pi):
    """Particles interact uniformly up to a hard cut off"""
    # TODO test for one particle.
    return np.less(x_i, L / 10, dtype=float)


def Garnier(x_i, L=2 * np.pi):
    """Interaction function of Garnier et al. (2019)"""
    assert L > 0, "Length L must be greater than 0"
    return (L / 2) * np.less(x_i, L / 10, dtype=float)


def gamma(x_i, gamma=1 / 10, L=2 * np.pi):
    """ Variable cutoff indicator interaction"""
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    assert L > 0, "Length L must be greater than 0"
    # x_i = np.where(x_i == 0, 0, x_i)
    # print("x_i={0:.16f}".format(x_i[0]), type(x_i[0]))
    # print(x_i[0] == 0)
    inter = 1.0 * np.less(x_i, gamma * L, dtype=float)
    # print("Less than gamma L?", inter_leq, type(inter_leq[0]))
    # inter_geq = 1.0 * np.greater(x_i, 0.0, dtype=float)
    # print("greater than zero?{:.14f}".format(inter_geq[0]), type(inter_geq[0]))
    return inter


def smoothed_indicator(x, a=0.5):
    """ An indicator function with a softer cutoff"""
    f = np.zeros(len(x))
    for i in range(len(x)):
        if a <= np.abs(x[i]) <= a + 1:
            f[i] = np.exp(1 / (x[i] ** 2 - (a + 1) ** 2)) / np.exp(
                1 / (a ** 2 - (a + 1) ** 2)
            )
        elif np.abs(x[i]) < a:
            f[i] = 1
        else:
            f[i] = 0
    return f


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.arange(0, np.pi, 0.01)
    for function_str in dir():
        phi_function = eval(function_str)
        if callable(phi_function):
            plt.plot(x, phi_function(x), label=phi_function.__name__)
    plt.legend()
    plt.suptitle("Interaction Functions")
    plt.show()
