from numba import jit  # type: ignore
import numpy as np  # type: ignore
from typing import Callable, Dict, Generator, Tuple, Union
import warnings

import particle.herdingfunctions as Gs
import particle.interactionfunctions as phis


def set_initial_conditions(
    *,
    initial_dist_x: Union[str, np.ndarray] = None,
    initial_dist_v: Union[str, np.ndarray] = None,
    particle_count: int = 50,
    L: float = 2 * np.pi,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Sets initial conditions of the particle system from dictionary or np.array.

        Args:
            initial_dist_x: str from the initial condition dictionary or array-like list
                of positions.
            initial_dist_v: str from the initial condition dictionary or array-like list
                of velocities.
            particle_count: number of particle_count to simulate.
            length: length of the domain.
            **kwargs: optional keyword arguments for distributions.

        Returns:
            Tuple[np.ndarray]: the initial conditions.
      """
    assert L > 0 and np.issubdtype(
        type(L), np.number
    ), "Length must be a number greater than 0"

    if isinstance(initial_dist_x, str):
        position_initial_conditions = build_position_initial_condition(particle_count)
        try:
            x0 = position_initial_conditions[initial_dist_x]
            if len(x0) != particle_count:
                warnings.warn(
                    f"The {initial_dist_x} initial condition cannot be made into the right"
                    + f" length {particle_count}, only {len(x0)} particles will be"
                    + " simulated (if using 2NN setup, check particle count is a multiple of 3)"
                )
        except KeyError as error:
            print(
                f"{error} is not a valid keyword.",
                " Valid initial conditions for position are\n",
                f"{list(position_initial_conditions.keys())}",
            )

    elif isinstance(initial_dist_x, (list, tuple, np.ndarray)):
        print("Using array for position distribution")
        x0 = np.array(initial_dist_x)
        assert (
            len(x0) == particle_count
        ), f"The inputted array is not of length {particle_count}"

    elif initial_dist_x is None:
        print("No initial position condition, using uniform distribution \n")
        x0 = np.random.uniform(low=0, high=L, size=particle_count)

    else:
        raise TypeError("Initial_dist_x was not string or array-like")

    # Try to get arrays from dictionary
    if isinstance(initial_dist_v, str):
        velocity_initial_conditions = build_velocity_initial_condition(particle_count)
        try:
            v0 = velocity_initial_conditions[initial_dist_v]
            if len(v0) != particle_count:
                warnings.warn(
                    f"The {initial_dist_v} initial condition cannot be made into the right"
                    + f"length {particle_count}, only {len(v0)} particles will be"
                    + "simulated (if using 2NN setup, check particle count is a multiple of 3)"
                )
        except (KeyError) as error:
            print(
                f"{error} is not a valid keyword. Valid initial conditions for"
                f" velocity are {list(velocity_initial_conditions.keys())}"
            )

    elif isinstance(initial_dist_v, (list, tuple, np.ndarray)):
        print("Using array for velocity distribution")
        v0 = np.array(initial_dist_v)
        assert (
            len(v0) == particle_count
        ), f"The inputted array is not of length {particle_count}"

    elif initial_dist_v is None:
        print("No initial velocity condition, using positive normal distrbution\n")
        v0 = np.random.normal(loc=1, scale=2, size=particle_count)
    else:
        raise TypeError("Initial_dist_v  was not string or array-like")

    # Check that lengths of position and velocity are the same
    if len(x0) != len(v0):
        warnings.warn("Initial conditions are not of equal length, truncating...")
        v0 = v0[: len(x0)]
        x0 = x0[: len(v0)]
        print(len(x0), len(v0))

    return x0.astype(np.float64), v0.astype(np.float64)


def build_position_initial_condition(
    particle_count: int, L: float = 2 * np.pi
) -> Dict[str, np.ndarray]:
    """Builds dictionary of possible initial position conditions

        Args:
            particle_count: number of particles to simulate.
            L: length of the domain.

        Returns:
            Dict: keys are names of conditions, values are corresponding nparrays.
    """

    def _cluster(particle_count: int, loc: float, width: float) -> np.ndarray:
        """Helper function for easier cluster building"""
        cluster = np.random.uniform(
            low=loc - width / 2, high=loc + width / 2, size=particle_count
        )
        return cluster

    left_cluster = _cluster(
        particle_count=(2 * particle_count) // 3, loc=0, width=np.pi / 5
    )
    right_cluster = _cluster(
        particle_count=particle_count // 3, loc=np.pi, width=np.pi / 5
    )

    prog_spaced = np.array([0.5 * (n + 1) * (n + 2) for n in range(particle_count)])
    prog_spaced /= prog_spaced[-1]
    prog_spaced *= 2 * np.pi

    even_spaced = np.arange(0, L, L / particle_count)

    def build_clusters(
        particle_count: int = 200,
        number_of_clusters: int = 1,
        width: float = np.pi / 5,
        loc: float = 0,
    ) -> np.ndarray:
        """Build clustered position initial condition
            Create array describing position initial condition with `number_of_clusters`
            clusters evenly spaced with one fixed at `loc`. The width of the clusters is
            equal and `width`. There are no checks to see whether the clusters overlap, you
            have been warned.

            If the number of particles does not divide evenly into the number of clusters,
            the remaining particles will be added to the final cluster
         """

        cluster_array = []  # np.zeros(particle_count, dtype=float)
        cluster_size = particle_count // number_of_clusters
        separation = 2 * np.pi / number_of_clusters

        if separation <= width:
            warnings.warn("Clusters will overlap")

        for n in range(number_of_clusters):
            cluster = _cluster(
                particle_count=cluster_size,
                loc=(loc + n * separation) % (2 * np.pi),
                width=width,
            )
            cluster_array = np.hstack((cluster_array, cluster))

        while len(cluster_array) != particle_count:
            warnings.warn("Extending final cluster to correct particle_count")
            cluster_array = np.hstack((cluster_array, cluster_array[-1]))
        return cluster_array

    one_cluster = build_clusters(
        particle_count=particle_count, number_of_clusters=1, width=np.pi / 5, loc=0,
    )
    two_clusters = build_clusters(
        particle_count=particle_count, number_of_clusters=2, width=np.pi / 5, loc=0,
    )

    three_clusters = build_clusters(
        particle_count=particle_count, number_of_clusters=3, width=np.pi / 5, loc=0,
    )
    four_clusters = build_clusters(
        particle_count=particle_count, number_of_clusters=4, width=np.pi / 5, loc=0,
    )
    position_initial_conditions = {
        "uniform_dn": np.random.uniform(low=0, high=L, size=particle_count),
        "two_clusters_2N_N": np.concatenate((left_cluster, right_cluster)),
        "bottom_cluster": _cluster(
            particle_count=particle_count, loc=np.pi, width=np.pi / 5
        ),
        "top_cluster": _cluster(
            particle_count=particle_count, loc=0.0, width=np.pi / 5
        ),
        "even_spaced": even_spaced,
        "prog_spaced": prog_spaced,
        "one_cluster": one_cluster,
        "two_clusters": two_clusters,
        "three_clusters": three_clusters,
        "four_clusters": four_clusters,
    }

    return position_initial_conditions


def build_velocity_initial_condition(particle_count: int) -> Dict[str, np.ndarray]:
    """Builds dictionary of possible initial velocity conditions

        Args:
            particle_count: number of particles to simulate.

        Returns:
            Dict: keys are names of conditions, values are corresponding nparrays.
    """
    slower_pos = np.random.uniform(low=0, high=1, size=(2 * particle_count) // 3)
    faster_pos = np.random.uniform(low=1, high=2, size=(particle_count // 3))

    left_NN_cluster = -0.2 * np.ones(2 * particle_count // 3)
    right_N_cluster = 1.8 * np.ones(particle_count // 3)

    normal_left_NN_cluster = -0.2 + np.random.normal(
        scale=0.5, size=2 * particle_count // 3
    )
    normal_right_N_cluster = 1.8 + np.random.normal(scale=0.5, size=particle_count // 3)

    left_NN_cluster_0 = -0.45 * np.ones(2 * particle_count // 3)
    right_N_cluster_0 = 0.9 * np.ones(particle_count // 3)

    velocity_initial_conditions = {
        "pos_normal_dn": np.random.normal(
            loc=0.2, scale=np.sqrt(2), size=particle_count
        ),
        "neg_normal_dn": np.random.normal(
            loc=-0.2, scale=np.sqrt(2), size=particle_count
        ),
        "uniform_dn": np.random.uniform(low=0, high=1, size=particle_count),
        "pos_gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=particle_count),
        "neg_gamma_dn": -np.random.gamma(shape=7.5, scale=1.0, size=particle_count),
        "pos_const_near_0": 0.2 * np.ones(particle_count),
        "neg_const_near_0": -0.2 * np.ones(particle_count),
        "pos_const": 1.8 * np.ones(particle_count),
        "neg_const": -1.8 * np.ones(particle_count),
        "2N_N_cluster_const": np.concatenate((left_NN_cluster, right_N_cluster)),
        "2N_N_cluster_normal": np.concatenate(
            (normal_left_NN_cluster, normal_right_N_cluster)
        ),
        "2N_N_cluster_avg_0": np.concatenate((left_NN_cluster_0, right_N_cluster_0)),
    }
    return velocity_initial_conditions


def get_interaction_functions(
    interaction_function: str, herding_function: str,
):

    interaction_functions = {
        "Garnier": phis.Garnier,
        "Uniform": phis.uniform,
        "Zero": phis.zero,
        "Indicator": phis.indicator,
        "Smoothed Indicator": phis.smoothed_indicator,
        "Gamma": phis.gamma,
        "Normalised Gamma": phis.normalised_gamma,
        "Gaussian": phis.gaussian,
    }
    try:
        phi = interaction_functions[interaction_function]
    except KeyError as error:
        print(
            f"{error} is not valid."
            f" Valid interactions are {list(interaction_functions.keys())}"
        )

    # Get herding function from dictionary, if not valid, throw error
    herding_functions = {
        "Garnier": Gs.Garnier,
        "Hyperbola": Gs.hyperbola,
        "Smooth": Gs.smooth,
        "Alpha Smooth": Gs.alpha_smooth,
        "Step": Gs.step,
        "Symmetric": Gs.symmetric,
        "Zero": Gs.zero,
    }

    try:
        G = herding_functions[herding_function]

    except KeyError as error:
        print(
            f"{error} is not valid."
            f" Valid herding functions are {list(herding_functions.keys())}"
        )
    return phi, G


def get_trajectories(
    initial_dist_x: Union[str, np.ndarray] = None,
    initial_dist_v: Union[str, np.ndarray] = None,
    particle_count: int = 100,
    T_end: float = 10,
    dt: float = 0.1,
    L: float = 2 * np.pi,
    D: float = 0.0,
    G: Callable[[np.ndarray], np.ndarray] = Gs.smooth,
    phi: Callable[[np.ndarray], np.ndarray] = phis.zero,
    option: str = "numpy",
    scaling: str = "Local",
    gamma: float = 0.1,
    alpha: float = 1,
    record_time: float = 0.5,
):

    # Number of steps
    N = np.int64(T_end / dt)
    # Preallocate matrices
    # Size chosen too large then excess trimmed after (depends on save timestep)
    x_history = np.zeros(
        (np.int64(T_end / record_time), particle_count), dtype=np.float64
    )
    out_times = np.zeros(np.int64(T_end / record_time), dtype=np.float64)
    v_history = np.zeros_like(x_history)
    x, v = set_initial_conditions(
        initial_dist_x=initial_dist_x,
        initial_dist_v=initial_dist_v,
        particle_count=particle_count,
        L=L,
    )

    phi, G = get_interaction_functions(interaction_function=phi, herding_function=G)
    if option.lower() == "numpy" and scaling.lower() == "local":
        step = numpy_step
        calculate_interaction = calculate_local_interaction
    elif option.lower() == "numpy" and scaling.lower() == "global":
        step = numpy_step
        calculate_interaction = calculate_global_interaction

    elif option.lower() == "numba" and scaling.lower() == "local":
        step = numba_step
        G = jit(nopython=True)(G)
        phi = jit(nopython=True)(phi)
        calculate_interaction = jit(nopython=True)(calculate_local_interaction)
    elif option.lower() == "numba" and scaling.lower() == "global":
        step = numba_step
        G = jit(nopython=True)(G)
        phi = jit(nopython=True)(phi)
        calculate_interaction = jit(nopython=True)(calculate_global_interaction)

    else:
        raise ValueError(
            "Option must be numpy or numba, scaling must be global or local"
        )
    self_interaction = np.array(phi(0.0, L, gamma), dtype=np.float64)
    time_since_record = 0
    store_index = 0
    for n in range(N):
        x, v = next(
            step(
                x,
                v,
                D,
                dt,
                particle_count,
                L,
                G,
                phi,
                calculate_interaction,
                self_interaction,
                gamma,
                alpha,
            )
        )
        time_since_record += dt
        if time_since_record > record_time:
            # TODO: Change so that number of save points is input (and less!)
            x_history[store_index, :] = x
            v_history[store_index, :] = v
            out_times[store_index] = (n + 1) * dt
            store_index += 1
            time_since_record = 0

    used_rows = len(np.trim_zeros(v_history[:, 0], trim="b"))
    out_times = out_times[:used_rows]
    x_history = x_history[:used_rows, :]
    v_history = v_history[:used_rows, :]
    return out_times, x_history, v_history


def numpy_step(
    x: np.ndarray,
    v: np.ndarray,
    D: float,
    dt: float,
    particle_count: int,
    L: float,
    G: Callable[[np.ndarray], np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
    calculate_interaction: Callable,
    self_interaction: float = 1.0,
    gamma: float = 0.1,
    alpha: float = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields updated positions and velocites after one step using the Euler-Maruyama
    scheme to discretise the SDE.
    """
    noise_scale = np.sqrt(2 * D * dt)
    while 1:
        interaction = calculate_interaction(x, v, phi, self_interaction, L, gamma)
        x = (x + v * dt) % L  # Restrict to torus
        v = (
            v
            + (G(interaction, alpha) - v) * dt
            + noise_scale * np.random.normal(size=particle_count)
        )
        yield x, v


@jit(nopython=True)
def numba_step(
    x: np.ndarray,
    v: np.ndarray,
    D: float,
    dt: float,
    particle_count: int,
    L: float,
    G: Callable[[np.ndarray], np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
    calculate_interaction: Callable,
    self_interaction: float = 1.0,
    gamma: float = 0.1,
    alpha: float = 1,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    noise_scale = np.sqrt(2 * D * dt)
    while 1:
        interaction = calculate_interaction(x, v, phi, self_interaction, L, gamma)
        interaction = G(interaction, alpha)
        for particle in range(len(x)):
            x[particle] = (x[particle] + v[particle] * dt) % L  # Restrict to torus
            v[particle] = (
                v[particle]
                + (interaction[particle] - v[particle]) * dt
                + noise_scale * np.random.normal()
            )
        yield x, v


def calculate_local_interaction(
    x: np.ndarray,
    v: np.ndarray,
    phi: Callable[[np.ndarray], np.ndarray],
    self_interaction: float,
    L: float = 2 * np.pi,
    gamma: float = 0.1,
) -> np.ndarray:
    """Calculate local interaction term of the full particle system
        Args:
            x: np.array of current particle positions
            v: np.array of current particle velocities
            phi: interaction function
            L: domain length, float
        Returns:
            array: The calculated interaction at the current time step for each particle
        See Also:
            :py:mod:`~particle.interactionfunctions`
    """
    interaction_vector = np.zeros(len(x), dtype=np.float64)
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        particle_interaction = phi(np.minimum(distance, L - distance), L, gamma)
        weighted_avg = np.sum(v * particle_interaction) - v[particle] * self_interaction
        scaling = np.sum(particle_interaction) - self_interaction + 10 ** -15
        interaction_vector[particle] = weighted_avg / scaling
    return interaction_vector


def calculate_global_interaction(
    x: np.ndarray,
    v: np.ndarray,
    phi: Callable[[np.ndarray], np.ndarray],
    self_interaction: float,
    L: float,
    gamma: float = 0.1,
) -> np.ndarray:
    """Calculate global interaction term of the full particle system
        Args:
            x: np.array of current particle positions
            v: np.array of current particle velocities
            phi: interaction function
            L: domain length, float
        Returns:
            array: The calculated interaction at the current time step for each particle
        See Also:
            :py:mod:`~particle.interactionfunctions`
    """
    interaction_vector = np.zeros(len(x), dtype=np.float64)
    scaling = len(x) - 1 + 10 ** -15
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        particle_interaction = phi(np.minimum(distance, L - distance), L, gamma)
        weighted_avg = np.sum(v * particle_interaction) - v[particle] * self_interaction
        interaction_vector[particle] = weighted_avg / scaling
    return interaction_vector


def compare_methods_particle_count(
    particles: int = 100, T_end: float = 100, runs: int = 25
) -> None:
    from datetime import datetime

    # Run once without timing to compile
    x, v = get_trajectories(
        initial_dist_x="prog_spaced",
        initial_dist_v="pos_normal_dn",
        particle_count=particles,
        T_end=T_end,
        dt=0.1,
        L=2 * np.pi,
        D=0.5,
        G="Smooth",
        phi="Gamma",
        option="numba",
        gamma=0.3,
    )
    compiled_list = []
    numpy_list = []
    for particles in np.logspace(start=0, stop=4, num=25):
        start_compile = datetime.now()
        dt = 0.1
        x, v = get_trajectories(
            initial_dist_x="prog_spaced",
            initial_dist_v="pos_normal_dn",
            particle_count=np.int(particles),
            T_end=T_end,
            dt=dt,
            L=2 * np.pi,
            D=0.5,
            G="Smooth",
            phi="Gamma",
            option="numba",
            gamma=0.3,
        )
        compile_time = datetime.now() - start_compile
        start_numpy = datetime.now()
        x, v = get_trajectories(
            initial_dist_x="prog_spaced",
            initial_dist_v="pos_normal_dn",
            particle_count=np.int(particles),
            T_end=T_end,
            dt=dt,
            L=2 * np.pi,
            D=0.5,
            G="Smooth",
            phi="Gamma",
            option="numpy",
            gamma=0.3,
        )
        numpy_time = datetime.now() - start_numpy

        compiled_list.append(compile_time.total_seconds())
        print(f"{int(particles)} compiled particles ran in {compile_time}")
        numpy_list.append(numpy_time.total_seconds())
        print(f"{int(particles)} numpy particles ran in {numpy_time}")

    return numpy_list, compiled_list


def compare_methods_T_end(
    particles: int = 200, T_end: float = 100, runs: int = 25
) -> None:
    from datetime import datetime

    # Run once without timing to compile
    x, v = get_trajectories(
        initial_dist_x="prog_spaced",
        initial_dist_v="pos_normal_dn",
        particle_count=particles,
        T_end=T_end,
        dt=0.1,
        L=2 * np.pi,
        D=0.5,
        G="Smooth",
        phi="Gamma",
        option="numba",
        gamma=0.3,
    )
    compiled_list = []
    numpy_list = []
    for T in np.logspace(start=0, stop=5, num=30):
        start_compile = datetime.now()
        dt = 0.1
        x, v = get_trajectories(
            initial_dist_x="prog_spaced",
            initial_dist_v="pos_normal_dn",
            particle_count=200,
            T_end=T,
            dt=dt,
            L=2 * np.pi,
            D=0.5,
            G="Smooth",
            phi="Gamma",
            option="numba",
            gamma=0.3,
        )
        compile_time = datetime.now() - start_compile
        start_numpy = datetime.now()
        x, v = get_trajectories(
            initial_dist_x="prog_spaced",
            initial_dist_v="pos_normal_dn",
            particle_count=200,
            T_end=T,
            dt=dt,
            L=2 * np.pi,
            D=0.5,
            G="Smooth",
            phi="Gamma",
            option="numpy",
            gamma=0.3,
        )
        numpy_time = datetime.now() - start_numpy

        compiled_list.append(compile_time.total_seconds())
        print(f"{T} sim time compiled particles ran in {compile_time}")
        numpy_list.append(numpy_time.total_seconds())
        print(f"{T} sim time numpy particles ran in {numpy_time}")

    return numpy_list, compiled_list


def main(option: str, scaling: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 0.005
    T_end = 100
    t, x, v = get_trajectories(
        initial_dist_x="uniform_dn",
        initial_dist_v="pos_normal_dn",
        particle_count=100,
        T_end=T_end,
        dt=dt,
        L=2 * np.pi,
        D=0.5,
        G="Alpha Smooth",
        phi="Gamma",
        option=option,
        scaling=scaling,
        gamma=0.1,
        alpha=1,
    )
    # t = np.arange(0, T_end, dt)
    return t, x, v


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # type: ignore
    import particle.plotting as plotting
    import scipy.stats as stats

    t, x, v = main(option="numba", scaling="local")

    # print(x.shape)
    # print(x)
    # print(v.shape)
    # print(v)
    # numpy_list, compiled_list = compare_methods_particle_count(particles=20, T_end=100, runs=5)
    # numpy_list, compiled_list = compare_methods_T_end(particles=200, T_end=100, runs=5)

    # particle_counts = np.logspace(start=0, stop=5, num=30)
    #
    # fig, ax = plt.subplots()
    # ax.semilogx(particle_counts, numpy_list, label="Numpy")
    # ax.semilogx(particle_counts, compiled_list, label="Compiled")
    # ax.legend()
    # fig1, ax1 = plt.subplots()
    # ax1.plot(particle_counts, numpy_list, label="Numpy")
    # ax1.plot(particle_counts, compiled_list, label="Compiled")
    # ax1.legend()
    # plt.show()
    plt.hist(
        x.flatten(),
        bins=np.arange(x.min(), x.max(), np.pi / 30),
        density=True,
        label="Position",
    )
    plt.hist(
        v.flatten(),
        bins=np.arange(v.min(), v.max(), (v.max() - v.min()) / 30),
        density=True,
        label="Velocity",
    )
    mu_v = 1
    sigma = np.sqrt(0.5)
    _v = np.arange(mu_v - 5 * sigma, mu_v + 5 * sigma, 0.01)
    pde_stationary_dist = stats.norm.pdf(_v, mu_v, sigma)
    plt.plot(_v, pde_stationary_dist, label=r"Stationary D$^{\mathrm{n}}$")
    ani = plotting.anim_torus(t, x, v, variance=0.5)
    plt.show()
