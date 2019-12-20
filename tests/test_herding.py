import numpy as np

# import matplotlib.pyplot as plt
import particle.herdingfunctions as Gs


def test_odd():
    x = np.arange(-100, 100, 0.01)
    for i in dir(Gs):
        G = getattr(Gs, i)
        if callable(G):
            if G.__name__ == "Garnier":
                for h in np.arange(4.1, 10, 0.1):
                    assert (
                        G(x, h).all() == (-G(-x, h)).all()
                    ), "Function G.{} is not odd when h is {}!".format(G.__name__, h)
            elif G.__name__ == "step":
                for beta in np.arange(0, 100, 0.1):
                    assert (
                        G(x, beta).all() == (-G(-x, beta)).all()
                    ), "Function G.{} is not odd when h is {}!".format(G.__name__, h)
            else:
                assert (
                    G(x).all() == (-G(-x)).all()
                ), "Function G.{} is not odd when beta is {}!".format(G.__name__, h)


def test_intersects():
    x = np.arange(-10, 10, 0.01)
    for i in dir(Gs):
        G = getattr(Gs, i)
        if callable(G):
            if G.__name__ == "Garnier":
                for h in np.arange(4.1, 10, 0.1):
                    assert any(
                        np.isclose(G(x), x)
                    ), "Function G.{} does not intersect x when h is {}!".format(
                        G.__name__, h
                    )
            elif G.__name__ == "step":
                for beta in np.arange(0, 100, 0.1):
                    assert any(
                        np.isclose(G(x), x)
                    ), "Function G.{} does not intersect x when beta is {}!".format(
                        G.__name__, h
                    )
            else:
                assert any(
                    np.isclose(G(x), x)
                ), "Function G.{} does not intersect x!".format(G.__name__, h)


if __name__ == "__main__":
    test_odd()
    test_intersects()
