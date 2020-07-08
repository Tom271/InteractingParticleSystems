import numpy as np  # type: ignore

# Define interaction functions


def zero(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1) -> np.ndarray:
    """No interaction between particles"""
    return np.zeros_like(x_i)


def uniform(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1) -> np.ndarray:
    """All particles interact with every other particle equally"""
    return np.ones_like(x_i)


def indicator(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1) -> np.ndarray:
    """Particles interact uniformly up to a hard cut off"""
    # TODO test for one particle.
    return np.less(x_i, L / 10)


def gaussian(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1):
    """Particles always interact, strength given by Gaussian"""
    return np.exp(-(x_i ** 2) / 4)


def bump(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1):
    """Periodic bump function with plateau length a """
    a = 1 / 2
    bump = np.exp(1 / (1 - (a / np.pi) ** 2) - 1 / (1 - (x_i / np.pi) ** 2))
    return np.where(np.abs(x_i) <= a, 1.0, bump)


def Garnier(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1) -> np.ndarray:
    """Interaction function of Garnier et al. (2019)"""
    assert L > 0, "Length L must be greater than 0"
    return (L / 2) * np.less(x_i, L / 10, dtype=float)


def gamma(x_i: np.ndarray, L: float = 2 * np.pi, gamma: float = 0.1) -> np.ndarray:
    """ Variable cutoff indicator interaction"""
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    return 1.0 * np.less(x_i, gamma * L)


def normalised_gamma(x_i: np.ndarray, L: float = 2 * np.pi, **parameters) -> np.ndarray:
    """ Variable cutoff indicator interaction"""
    # gamma controls how much of the torus is seen and scales strength accordingly.
    # gamma = 0.1 corresponds to phi_Garnier, gamma=0 is phi_zero
    # and gamma = 1 is phi_one
    # assert L > 0, "Length L must be greater than 0"
    gamma = parameters.get("gamma", 0.1)

    if gamma != 0.0:
        return 1 / (2 * gamma) * np.less(x_i, gamma * L, dtype=np.float)
    else:
        return np.zeros_like(x_i)


def smoothed_indicator(x: np.ndarray, L: float = 2 * np.pi, **parameters) -> np.ndarray:
    """ An indicator function with a softer cutoff"""
    a = parameters.get("a", 2)
    f = np.zeros(len(x))
    f[a <= np.abs(x) <= a + 1] = np.exp(
        1 / (x ** 2 - (a + 1) ** 2) - 1 / (a ** 2 - (a + 1) ** 2)
    )
    f[np.abs(x) < a] = 1
    return f


if __name__ == "__main__":
    from matplotlib import rc
    import matplotlib.pyplot as plt
    import seaborn as sns

    # rc("text", usetex=True)
    sns.set(style="white", context="talk")

    x = np.arange(-np.pi, np.pi, 0.01)
    # for function_str in dir():
    #     phi_function = eval(function_str)
    #     if callable(phi_function):
    #         plt.plot(x, phi_function(x), label=phi_function.__name__)
    plt.plot(x, bump(x, L=2 * np.pi))

    # plt.legend()
    plt.suptitle("Interaction Functions")
    plt.show()
