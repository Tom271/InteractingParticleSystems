import numpy as np
from math import atan
from scipy.integrate import simps


def M1_hom(f_t, v):
    dv = v[1] - v[0]
    # Simpson's Rule
    avg_v = simps(v * f_t, dx=dv)
    return avg_v


def Mn(f_t, v, n):
    dv = v[1] - v[0]
    # Simpson's Rule
    avg_v = simps((v ** n) * f_t, dx=dv)

    return avg_v


if __name__ == "__main__":
    # TODO Put tests here, plots of functions?
    y = "Hello World"
    print(y)
