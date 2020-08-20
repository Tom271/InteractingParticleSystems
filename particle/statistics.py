from matplotlib import cycler
import matplotlib.pyplot as plt
import numpy as np


def calculate_avg_vel(t, x, v):
    return v.mean(axis=1)


def calculate_variance(t, x, v):
    return v.var(axis=1)


def moving_average(a: np.ndarray, n: int = 3) -> np.ndarray:
    """Calculate moving average of an array"""
    ret = np.cumsum(a, dtype=np.float64)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1 :] / n


def Q_order_t(
    x: np.ndarray, gamma_tilde: float = 0.1, L: float = 2 * np.pi
) -> np.ndarray:
    """ Calculate order parameter from position array"""
    Q_vector = np.empty(len(x))
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        distance = np.minimum(distance, L - distance)
        Q_vector[particle] = np.sum(distance < L * gamma_tilde)
    Q = (1 / (len(x)) ** 2) * np.sum(Q_vector)
    return Q


def CL2(x: np.ndarray, L: float = (2 * np.pi)) -> np.ndarray:
    """Calculates the centered L2 discrepancy

    Uses the position vector to calculate the squared centred L2 discrepancy at the
    current time point for quantifying uniformity of particle distribution.

    Args:
        x (array_like): Particle positions at a given time.
        L (float): domain length, must be greater than zero.

    Returns:
        float: the CL2 discrepancy at this time.

    Adapted from `Stack Overflow <https://stackoverflow.com/questions/50364048/\
python-removing-multiple-for-loops-for-faster-calculation-centered-l2-discrepa>`_
    """
    N = len(x)
    term3 = 0
    term2 = np.sum(2.0 + np.abs(x / L - 0.5) - np.abs(x / L - 0.5) ** 2)
    for i in range(N):
        term3 += np.sum(
            1.0
            + np.abs(x[i] / L - 0.5) / 2
            + np.abs(x / L - 0.5) / 2
            - np.abs(x[i] / L - x / L) / 2
        )
    CL2 = (13 / 12) - (term2 - term3 / N) / N

    return CL2


def calculate_l1_convergence(
    t, x, v, plot_hist: bool = False, final_plot_time: float = 100000,
):
    """Calculate l1 error between positions and uniform distribution

    Load data from file name and calculate the l1 discrepancy from a uniform
    distribution on the torus. Can also plot the histogram of the position
    density over time.
    """

    dt = t[1] - t[0]
    error = []
    if plot_hist is True:
        colormap = plt.get_cmap("viridis")
        fig, ax = plt.subplots()
        ax.set_prop_cycle(
            cycler(color=[colormap(k) for k in np.linspace(1, 0, int(1 / dt))])
        )
    for i in np.arange(0, int(min(len(t), final_plot_time // 0.5))):
        hist_x, bin_edges = np.histogram(
            x[i, :], bins=np.arange(0, 2 * np.pi, np.pi / 60), density=True
        )

        # hist_x = hist_x / len(x[0, :])
        error_t = np.abs((1 / (2 * np.pi)) - hist_x).sum()
        error.append(error_t)

        if plot_hist is True:
            ax.plot(bin_edges[:-1], hist_x)

    if plot_hist is True:
        ax.plot([0, 2 * np.pi], [1 / (2 * np.pi), 1 / (2 * np.pi)], "k--")
        ax.set(xlim=[0, 2 * np.pi], xlabel="Position", ylabel="Density")
        return error, fig, ax
    else:
        return error


def calculate_stopping_time(v: np.ndarray, dt: float):
    """Given a velocity trajectory matrix, calculate the time to convergence.
     """
    tol = 0.5e-2
    zero_mask = np.isclose(np.mean(v, axis=1), 0, atol=tol)
    one_mask = np.isclose(np.mean(v, axis=1), 1, atol=tol)
    neg_one_mask = np.isclose(np.mean(v, axis=1), -1, atol=tol)
    expect_converge_value = np.sign(np.mean(v[0, :]))
    conv_steps = [True for _ in range(int(2 / dt))]
    conv_steps.append(False)
    final_avg = np.mean(v[-1,])
    if np.isclose(final_avg, 1.0, atol=tol):
        mask = one_mask
    elif np.isclose(final_avg, -1.0, atol=tol):
        mask = neg_one_mask
    elif np.isclose(final_avg, 0.0, atol=tol):
        mask = zero_mask
    else:
        print("Did not converge to expected values, converged to {}".format(final_avg))
        mask = np.isclose(np.mean(v, axis=1), final_avg, atol=tol)
        tau = len(v[:, 0]) * dt
        return final_avg, tau

    count = 0
    n_more = iter(conv_steps)
    while not mask[count] or next(n_more):
        tau = count * dt
        count += 1
        if count >= len(mask):
            break
    return final_avg, tau
