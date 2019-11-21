import numpy as np
from particle import (
    calculate_interaction,
    run_full_particle_system,
    run_hom_particle_system,
    phi_zero,
    phi_uniform,
    no_G,
    step_G,
    Garnier_G,
    CL2,
)
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime

np.random.seed(1)
# INTERACTION CHECKS

L = 2 * np.pi
# interaction_vector = np.zeros(5)


# Does phi_zero return interaction zero?
def test_zeros():
    x = np.random.uniform(low=0, high=2 * np.pi, size=1000)
    v = np.random.uniform(low=-100, high=100, size=1000)
    assert calculate_interaction(x, v, phi_zero, L).all() == 0.0


# Does phi_one return same as average v?
def test_ones():
    x = np.random.uniform(low=0, high=2 * np.pi, size=1000)
    v = np.random.uniform(low=-100, high=100, size=1000)
    out = calculate_interaction(x, v, phi_uniform, L)
    print(out)
    print(np.mean(v))
    assert np.equal(out, np.mean(v)).all()


# TODO: Does it match hand calc for 5 particles?

# def test_hand():
#     #See pg81 in Notebook
# x = np.array([1, 6.2, 1, 0.5, 0.1])
# v = np.array([-1, 0.1, 2, 5, 1])
#     hand_calc = [16/5 ]
#     assert calculate_interaction(x,v, phi_indicator)[0] == hand_calc[0]


# CONVERGENCE CHECKS
# Does it converge to N(0,D) when phi=0?


def test_OU():
    particles = 20
    diffusion = 4
    startTime = datetime.now()
    t, x, v = run_full_particle_system(
        interaction_function="Zero",
        particles=particles,
        D=diffusion,
        initial_dist_v=np.random.normal(loc=0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
    )
    print(
        "Full system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    startTime = datetime.now()
    t_hom, v_hom = run_hom_particle_system(
        particles=particles,
        D=diffusion,
        initial_dist=np.random.normal(loc=0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
        G=no_G,
    )
    print(
        "Homogeneous system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    model_prob_x, _ = np.histogram(
        x[-500:-1,].flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True
    )
    model_prob_v, _ = np.histogram(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    # hom_model_prob_v, _ = np.histogram(v_hom[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True)
    # print(np.sum(np.abs(model_prob_v - hom_model_prob_v)))
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].hist(x.flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True)
    ax[0].plot([x.min(), x.max()], [1 / (2 * np.pi), 1 / (2 * np.pi)], "--")
    ax[0].set(xlabel="Position")

    ax[1].hist(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    ax[1].hist(
        v_hom[-500:-1,].flatten(),
        bins=np.arange(v.min(), v.max(), 0.15),
        density=True,
        alpha=0.3,
    )
    ax[1].plot(
        np.arange(-v.max(), v.max(), 0.01),
        stats.norm.pdf(
            np.arange(-v.max(), v.max(), 0.01), loc=0, scale=np.sqrt(diffusion)
        ),
        "--",
    )
    ax[1].set(xlabel="Velocity")
    # true_prob_x = 1 / (2 * np.pi) * np.ones(len(model_prob_x))
    true_prob_v = stats.norm.pdf(
        np.arange(v.min(), v.max() - 0.15, 0.15), loc=0, scale=np.sqrt(diffusion)
    )
    fig.savefig("xvhist.jpg", format="jpg", dpi=1000)
    print(
        "KL Divergence of velocity distribution:",
        stats.entropy(model_prob_v, true_prob_v),
    )

    print(
        "L2 discrepancy of space distribution:",
        CL2(x[-100:,].flatten()),
        ", expected is ",
        (1 / particles * (5 / 4 - 13 / 12)),
    )
    plt.show()


# Does it converge to N(\pm \xi) when phi=1?
def test_normal():
    particles = 20
    diffusion = 4
    startTime = datetime.now()
    t, x, v = run_full_particle_system(
        interaction_function="Uniform",
        herding_function="Step",
        particles=particles,
        D=diffusion,
        initial_dist_v=np.random.normal(loc=-0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
    )
    print(
        "Full system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    startTime = datetime.now()
    t_hom, v_hom = run_hom_particle_system(
        particles=particles,
        D=diffusion,
        initial_dist=np.random.normal(loc=-0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
        G=step_G,
    )
    print(
        "Homogeneous system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    model_prob_x, _ = np.histogram(
        x[-500:-1,].flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True
    )
    model_prob_v, _ = np.histogram(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    # hom_model_prob_v, _ = np.histogram(v_hom[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True)
    # print(np.sum(np.abs(model_prob_v - hom_model_prob_v)))
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].hist(x.flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True)
    ax[0].plot([x.min(), x.max()], [1 / (2 * np.pi), 1 / (2 * np.pi)], "--")
    ax[0].set(xlabel="Position")

    ax[1].hist(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    ax[1].hist(
        v_hom[-500:-1,].flatten(),
        bins=np.arange(v.min(), v.max(), 0.15),
        density=True,
        alpha=0.3,
    )
    ax[1].plot(
        np.arange(-v.max(), v.max(), 0.01),
        stats.norm.pdf(
            np.arange(-v.max(), v.max(), 0.01), loc=-1, scale=np.sqrt(diffusion)
        ),
        "--",
    )
    ax[1].set(xlabel="Velocity")

    # true_prob_x = 1 / (2 * np.pi) * np.ones(len(model_prob_x))
    true_prob_v = stats.norm.pdf(
        np.arange(v.min(), v.max() - 0.15, 0.15), loc=1, scale=np.sqrt(diffusion)
    )
    fig.savefig("xvhistmeanone.jpg", format="jpg", dpi=1000)
    print(
        "KL Divergence of velocity distribution:",
        stats.entropy(model_prob_v, true_prob_v),
    )

    print(
        "L2 discrepancy of space distribution:",
        CL2(x[-20:,].flatten()),
        ", expected is ",
        (1 / particles * (5 / 4 - 13 / 12)),
    )
    plt.show()


def test_Garnier():
    particles = 20
    diffusion = 4
    well_depth = 6
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    startTime = datetime.now()
    t, x, v = run_full_particle_system(
        interaction_function="Garnier",
        herding_function="Garnier",
        particles=particles,
        D=diffusion,
        initial_dist_v=np.random.normal(loc=0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
        well_depth=well_depth,
        L=10,
    )
    print(
        "Full system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    startTime = datetime.now()
    t_hom, v_hom = run_hom_particle_system(
        particles=particles,
        D=diffusion,
        initial_dist=np.random.normal(loc=0.5, scale=np.sqrt(2), size=particles),
        T_end=100,
        G=Garnier_G,
        well_depth=well_depth,
    )
    print(
        "Homogeneous system solved\n",
        "Time to solve was  {} seconds".format(datetime.now() - startTime),
    )
    model_prob_x, _ = np.histogram(
        x[-500:-1,].flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True
    )
    model_prob_v, _ = np.histogram(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    # hom_model_prob_v, _ = np.histogram(v_hom[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True)
    # print(np.sum(np.abs(model_prob_v - hom_model_prob_v)))
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].hist(x.flatten(), bins=np.arange(x.min(), x.max(), 0.15), density=True)
    ax[0].plot([x.min(), x.max()], [1 / (2 * np.pi), 1 / (2 * np.pi)], "--")
    ax[0].set(xlabel="Position")

    ax[1].hist(
        v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15), density=True
    )
    ax[1].hist(
        v_hom[-500:-1,].flatten(),
        bins=np.arange(v.min(), v.max(), 0.15),
        density=True,
        alpha=0.3,
    )
    ax[1].plot(
        np.arange(-v.max(), v.max(), 0.01),
        stats.norm.pdf(
            np.arange(-v.max(), v.max(), 0.01), loc=xi, scale=np.sqrt(diffusion)
        ),
        "--",
    )
    ax[1].set(xlabel="Velocity")
    # true_prob_x = 1 / (2 * np.pi) * np.ones(len(model_prob_x))
    true_prob_v = stats.norm.pdf(
        np.arange(v.min(), v.max() - 0.15, 0.15), loc=1, scale=np.sqrt(diffusion)
    )
    fig.savefig("xvhistmeanone.jpg", format="jpg", dpi=1000)
    print(
        "KL Divergence of velocity distribution:",
        stats.entropy(model_prob_v, true_prob_v),
    )

    print(
        "L2 discrepancy of space distribution:",
        CL2(x[-100:,].flatten()),
        ", expected is ",
        (1 / particles * (5 / 4 - 13 / 12)),
    )
    plt.show()


def test_CL2():
    N = 500
    trials = 500
    data = np.zeros(trials)
    L = 10
    for i in range(trials):
        x = np.random.uniform(low=0, high=L, size=(N, 1))
        # x = np.random.normal(loc=0, scale=1, size=(N,1))
        data[i] = CL2(x, L)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(np.arange(0, trials), data)
    ax[0].plot([0, trials], [(1 / N * (5 / 4 - 13 / 12)), 1 / N * (5 / 4 - 13 / 12)])
    ax[0].plot([0, trials], [np.mean(data), np.mean(data)], "--")

    ax[1].hist(data, density=True)
    print(np.mean(data), np.var(data))
    print("Expected:", (1 / N * (5 / 4 - 13 / 12)), 1 / N ** 2)
    plt.show()
    assert np.isclose(np.mean(data), (1 / N * (5 / 4 - 13 / 12)), atol=1e-4)
    assert np.isclose(np.var(data), 1 / N ** 2, atol=1e-4)


if __name__ == "__main__":
    test_zeros()
    test_ones()
    # test_normal()
    # test_OU()
    # test_CL2()
    # test_Garnier()
