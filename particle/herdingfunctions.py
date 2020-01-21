import numpy as np


def zero(u):
    """ No herding occurs """
    return np.zeros_like(u)


def step(u, beta=1):
    """ Discontinuous herding function """
    assert beta >= 0, "Beta must be greater than 0"
    return (u + beta * np.sign(u)) / (1 + beta)


def smooth(u):
    """ Smooth herding function"""
    return np.arctan(u, dtype=float) / np.arctan(1.0, dtype=float)


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
