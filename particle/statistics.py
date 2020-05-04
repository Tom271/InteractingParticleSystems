import numpy as np


def avg_velocity(v):
    return np.mean(v, axis=1)


def Q_order_t(x, gamma_tilde=0.1, L=2 * np.pi):
    Q_vector = np.empty(len(x))
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        distance = np.minimum(distance, L - distance)
        Q_vector[particle] = np.sum(distance < L * gamma_tilde)
    Q = (1 / (len(x)) ** 2) * np.sum(Q_vector)
    return Q


def CL2(x, L=(2 * np.pi)):
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


def calculate_stopping_time(v: np.ndarray, dt: float) -> float:
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
