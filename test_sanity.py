import numpy as np
#INTERACTION CHECKS

def phi_zero(x_i_):
    return np.zeros_like(x_i_)

def phi_ones(x_i_):
    return np.ones_like(x_i_)

def phi_indicator(x_i_):
    out = np.array([float(i >= 0 and i <= 1) for i in x_i_])
    if len(out) == 1:
        return float(out[0])
    else:
        return out


def interaction(x,v,phi):
    interaction_vector = np.zeros(len(x))
    L = 2*np.pi
    for particle, position in enumerate(x):
        d1=np.abs(x - position)
        d2 =L-np.abs(x - position)
        dist = np.minimum(d1, d2)
        interaction = phi(dist)
        numerator = np.sum(v * interaction)
        #denom = np.sum(interaction)
        interaction_vector[particle] = numerator / len(x)
    return interaction_vector

x = np.random.uniform(low=0, high=2*np.pi, size=1000)
v = np.random.uniform(low=-100, high=100, size=1000)

#Does phi_zero return interaction zero?
def test_foo():
    def test_zeros():
        assert interaction(x,v, phi_zero).all() == np.zeros(len(x)).all()
    return None

#Does phi_one return same as average v?
def test_ones():
    assert interaction(x,v, phi_ones).all() == np.mean(v).all()

#Does it match hand calc for 5 particles?
x = np.array([1, 6.2, 1, 0.5, 0.1])
v = np.array([-1, 0.1, 2, 5, 1])

def test_hand():
    #See pg81 in Notebook
    hand_calc = [16/5 ]
    assert interaction(x,v, phi_indicator)[0] == hand_calc[0]



#MODEL CHECKS
#Does it converge to N(0,D) when phi=0?

#Does it converge to N(\pm \xi) when phi=1?
