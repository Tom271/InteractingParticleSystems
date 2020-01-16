import numpy as np
from particle.clssimulate_gen import HerdingFunction as G
import particle.interactionfunctions as phis

# Test the herding functions are odd, i.e. G(x) = -G(-x) for all x


def test_zero_odd():
    x = np.arange(-100, 100, 0.01)
    assert G.zero(x).all() == (-G.zero(-x)).all(), "Function G.zero is not odd!"


def test_step_odd():
    x = np.arange(-100, 100, 0.01)
    for beta in np.arange(0, 10, 0.01):
        assert (
            G.step(x, beta).all() == (-G.step(-x, beta)).all()
        ), "Function G.step is not odd when beta is {}!".format(beta)


def test_smooth_odd():
    x = np.arange(-100, 100, 0.01)
    assert G.smooth(x).all() == (-G.smooth(-x)).all(), "Function G.smooth is not odd!"


def test_Garnier_odd():
    x = np.arange(-100, 100, 0.01)
    for h in np.arange(4.1, 10, 0.1):
        assert (
            G.Garnier(x, h).all() == (-G.Garnier(-x, h)).all()
        ), "Function G.Garnier is not odd when h is {}!".format(h)


# Test that G(x) = x for some value of x


def test_zero_intersects():
    x = np.arange(-100, 100, 0.01)
    assert np.isclose(G.zero(x), x).any(), "Function G.zero does not intersect y=x!"


def test_step_intersects():
    x = np.arange(-100, 100, 0.01)
    for beta in np.arange(0, 100, 0.1):
        assert any(
            np.isclose(G.step(x), x)
        ), "Function G.step does not intersect x when beta is {}!".format(beta)


def test_smooth_intersects():
    x = np.arange(-100, 100, 0.01)
    assert np.isclose(G.smooth(x), x).any(), "Function G.smooth does not intersect y=x!"


def test_Garnier_intersects():
    x = np.arange(-100, 100, 0.01)
    for h in np.arange(4, 100, 0.1):
        assert any(
            np.isclose(G.Garnier(x), x)
        ), "Function G.Garnier does not intersect x when h is {}!".format(h)


def test_gamma_zero():
    x = np.random.uniform(low=0, high=2 * np.pi, size=10)
    v = np.random.uniform(low=-100, high=100, size=10)
    phi_gamma_x = phis.gamma(x, gamma=0.0, L=2 * np.pi)
    phi_zero_x = phis.zero(x)
    print(phi_gamma_x)
    print(phis.zero(x))
    print(G.smooth(phi_zero_x))
    print(G.smooth(phi_gamma_x))
    assert G.smooth(phi_gamma_x).all() == G.smooth(phi_zero_x).all(), "UhOh"


if __name__ == "__main__":
    test_gamma_zero()
    # test_odd()
    # test_Garnier_intersects()
