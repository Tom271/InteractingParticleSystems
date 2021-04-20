import numpy as np  # type: ignore


def zero(u: np.ndarray) -> np.ndarray:
    """ No herding occurs """
    return np.zeros_like(u)


def step(u: np.ndarray, beta: float = 1) -> np.ndarray:
    """ Discontinuous herding function """
    assert beta >= 0, "Beta must be greater than 0"
    return (u + beta * np.sign(u)) / (1 + beta)


def smooth(u: np.ndarray, alpha: float = 1) -> np.ndarray:
    """ Smooth herding function"""
    return np.arctan(u) / np.arctan(1.0)


def alpha_smooth(u: np.ndarray, alpha: float) -> np.ndarray:
    """ Smooth herding function"""
    return np.arctan(alpha * u) / np.arctan(alpha)


def hyperbola(u: np.ndarray) -> np.ndarray:
    """ Hyperbola herding function"""
    tol = 0.01
    herding = np.empty_like(u)
    herding[np.isclose(u, 0, atol=tol)] = 0
    herding[np.logical_not(np.isclose(u, 0, atol=tol))] = (
        1 / u[np.logical_not(np.isclose(u, 0, atol=tol))]
    )
    return herding


def symmetric(u: np.ndarray, alpha: float) -> np.ndarray:
    """ Herding function symmetric about 0 and 1
    Only symmetric until u=2.
     """
    herding = np.empty_like(u)
    herding[np.abs(u) <= 1] = (u[np.abs(u) <= 1] + np.sign(u[np.abs(u) <= 1])) / 2
    herding[np.abs(u) > 1] = (-u[np.abs(u) > 1] + 3 * np.sign(u[np.abs(u) > 1])) / 2
    return herding


def Garnier(u: np.ndarray, alpha: float = 6) -> np.ndarray:
    """ Herding function of Garnier et al. (2019)"""
    return (((alpha + 1) / 5) * u) - ((alpha / 125) * (u ** 3))


def main():
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


if __name__ == "__main__":
    # main()
    from matplotlib import rc
    import matplotlib.pyplot as plt
    import seaborn as sns

    rc("text", usetex=True)
    sns.set(style="white", context="talk")

    x = np.arange(-4, 4, 0.01)
    # for function_str in dir():
    #     phi_function = eval(function_str)
    #     if callable(phi_function):
    #         plt.plot(x, phi_function(x), label=phi_function.__name__)
    h = 10
    xi = 5 * np.sqrt((h - 4) / h)
    plt.plot(x, Garnier(x, h))
    plt.plot([xi, xi], [-4, 4])
    plt.plot(x, x, "--")
    # plt.legend()
    plt.suptitle("Herding  Function")
    plt.show()
