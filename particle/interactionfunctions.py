import numpy as np


# Define interaction functions
def zero(x_i_):
    return np.zeros_like(x_i_)


def uniform(x_i_):
    return np.ones_like(x_i_)


def indicator(x_i_):
    # TODO test for one particle.
    return 5 * np.less_equal(x_i_, 0.01, dtype=float)


def Garnier(x_i_, L=2 * np.pi):
    assert L > 0, "Length L must be greater than 0"
    return (L / 2) * np.less_equal(x_i_, L / 10, dtype=float)


def gamma(x_i_, gamma=1 / 10, L=2 * np.pi):
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    assert L > 0, "Length L must be greater than 0"
    return ((L * gamma) / 2) * np.less_equal(x_i_, gamma * L, dtype=float)


def smoothed_indicator(x, a):
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
