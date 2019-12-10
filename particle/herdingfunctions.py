import numpy as np


# Define herding functions
def step(u, beta=1):
    assert beta >= 0, "Beta must be greater than 0"
    return (u + beta * np.sign(u)) / (1 + beta)


def smooth(u):
    return np.arctan(u) / np.arctan(1)


def zero(u):
    return 0


def Garnier(u, h):
    return (((h + 1) / 5) * u) - ((h / 125) * (u ** 3))
