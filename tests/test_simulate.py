import matplotlib.pyplot as plt
import numpy as np
from particle.statistics import calculate_stopping_time
import seaborn as sns

sns.set()


def test_calculate_stopping_time():
    x = np.arange(0.1, 50, 0.01)
    conv_to_0 = np.exp(-x) + np.sin(np.pi * x) / (3 * x)
    conv_to_1 = 1 + np.exp(-x) + np.sin(np.pi * x) / (3 * x)
    conv_to_neg_1 = -1 + np.exp(-x) + np.sin(np.pi * x) / (3 * x)

    conv_to_0 = np.tile(conv_to_0, (10, 1))
    conv_to_1 = np.tile(conv_to_1, (10, 1))
    conv_to_neg_1 = np.tile(conv_to_neg_1, (10, 1))
    return conv_to_0.T, conv_to_1.T, conv_to_neg_1.T


if __name__ == "__main__":
    conv_to_0, conv_to_1, conv_to_neg_1 = test_calculate_stopping_time()
    dt = 0.01
    x = np.arange(0.1, 50, dt)

    tau_0 = calculate_stopping_time(conv_to_0, dt, 0)
    tau_1 = calculate_stopping_time(conv_to_1, dt, 1)
    tau_neg_1 = calculate_stopping_time(conv_to_neg_1, dt, -1)
    print("Time to hit 0 was:", tau_0)
    print("Time to hit 1 was:", tau_1)
    print("Time to hit -1 was:", tau_neg_1)
    plt.plot(x, conv_to_0[:, 0])
    plt.plot(x, conv_to_1[:, 0])
    plt.plot(x, conv_to_neg_1[:, 0])
    plt.plot([tau_0, tau_0], [-1, 1])
    plt.plot([tau_1, tau_1], [-1, 1])
    plt.plot([tau_neg_1, tau_neg_1], [-1, 1])

    plt.show()
