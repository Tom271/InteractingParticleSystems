import numpy as np
from numba import jit


@jit(nopython=True)
def zero(u):
    """ No herding occurs """
    return np.zeros_like(u)


@jit(nopython=True)
def step(u, beta=1):
    """ Discontinuous herding function """
    assert beta >= 0, "Beta must be greater than 0"
    return (u + beta * np.sign(u)) / (1 + beta)


@jit(nopython=True)
def smooth(u):
    """ Smooth herding function"""
    return np.arctan(u, dtype=float) / np.arctan(1.0, dtype=float)


def hyperbola(u):
    """ Hyperbola herding function"""
    tol = 0.01
    herding = np.empty_like(u)
    herding[np.isclose(u, 0, atol=tol)] = 0
    herding[np.logical_not(np.isclose(u, 0, atol=tol))] = (
        1 / u[np.logical_not(np.isclose(u, 0, atol=tol))]
    )
    return herding


def symmetric(u):
    """ Herding function symmetric about 0 and 1
    Only symmetric until u=2.
     """
    herding = np.empty_like(u)
    herding[abs(u) <= 1] = (u[abs(u) <= 1] + np.sign(u[abs(u) <= 1])) / 2
    herding[abs(u) > 1] = (-u[abs(u) > 1] + 3 * np.sign(u[abs(u) > 1])) / 2
    return herding


@jit(nopython=True)
def Garnier(u, h=6):
    """ Herding function of Garnier et al. (2019)"""
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set()

    x = np.arange(-4, 4, 0.01)
    for function_str in dir():
        G_function = eval(function_str)
        if callable(G_function):
            plt.plot(x, G_function(x), label=G_function.__name__)
    plt.plot(x, x, "k--", alpha=0.25, label="y=x")
    plt.legend()
    plt.suptitle("Herding Functions")
    plt.show()
