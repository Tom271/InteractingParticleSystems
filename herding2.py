import numpy as np
from math import atan
from scipy.integrate import simps

def phi_part(v):
    """Doc string"""
    avg_v = np.mean(v)
    return avg_v

def phi_pde(f_t, v):
    dv = v[1] - v[0]
    #Simpson's Rule
    avg_v =  simps(v*f_t, dx=dv)

    return avg_v

def step_G(u, beta=1):
    """Doc string"""
    assert beta >= 0 , 'Beta must be greater than 0'
    interaction = (u + beta * np.sign(u))/ (1 + beta)

    return interaction

def smooth_G(u, beta=None):
    interaction = atan(u)/atan(1)
    return interaction

def sigmoid_G(u, beta=None):
    print("Not implemented yet")
    interaction = atan(u)/atan(1)
    return interaction

def no_G(u, beta=None):
    return 0
if __name__ == "__main__":
    # Put tests here
    y = 'Hello World'
    print(y)
