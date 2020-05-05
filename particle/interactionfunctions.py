import numpy as np
from numba import jit


# Define interaction functions
@jit(nopython=True)
def zero(x_i):
    """No interaction between particles"""
    return np.zeros_like(x_i)


@jit(nopython=True)
def uniform(x_i):
    """All particles interact with every other particle equally"""
    return np.ones_like(x_i)


@jit(nopython=True)
def indicator(x_i, L=2 * np.pi):
    """Particles interact uniformly up to a hard cut off"""
    # TODO test for one particle.
    return np.less(x_i, L / 10, dtype=float)


@jit(nopython=True)
def Garnier(x_i, L=2 * np.pi):
    """Interaction function of Garnier et al. (2019)"""
    assert L > 0, "Length L must be greater than 0"
    return (L / 2) * np.less(x_i, L / 10, dtype=float)


@jit(nopython=True)
def gamma(x_i, gamma=1 / 10, L=2 * np.pi):
    """ Variable cutoff indicator interaction"""
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    assert L > 0, "Length L must be greater than 0"
    inter = 1.0 * np.less(x_i, gamma * L, dtype=float)
    return inter


@jit(nopython=True)
def normalised_gamma(x_i, gamma=1 / 10, L=2 * np.pi):
    """ Variable cutoff indicator interaction"""
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    # assert L > 0, "Length L must be greater than 0"
    if gamma != 0.0:
        inter = 1 / (2 * gamma) * np.less(x_i, gamma * L, dtype=float)
    else:
        inter = np.zeros_like(x_i)
    return inter


@jit(nopython=True)
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
