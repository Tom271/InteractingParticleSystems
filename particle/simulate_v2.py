from numba import jit  # type: ignore
import numpy as np  # type: ignore
from typing import Callable, Dict, Generator, Tuple, Union
import warnings

from particle.herdingfunctions import smooth
import particle.interactionfunctions as phis
import particle.plotting as plotting


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
    return x0, v0


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
    right_cluster = left_cluster + L / 2

    prog_spaced = np.array([0.5 * (n + 1) * (n + 2) for n in range(particle_count)])
    prog_spaced /= prog_spaced[-1]
    prog_spaced *= 2 * np.pi

    even_spaced = np.arange(0, L, L / particle_count)
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
            loc=1.2, scale=np.sqrt(2), size=particle_count
        ),
        "neg_normal_dn": np.random.normal(
            loc=-1.2, scale=np.sqrt(2), size=particle_count
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


def get_trajectories(
    initial_dist_x: Union[str, np.ndarray] = None,
    initial_dist_v: Union[str, np.ndarray] = None,
    particle_count: int = 100,
    T_end: float = 10,
    dt: float = 0.1,
    L: float = 2 * np.pi,
    D: float = 0.0,
    G: Callable[[np.ndarray], np.ndarray] = smooth,
    phi: Callable[[np.ndarray], np.ndarray] = phis.zero,
    option: str = "numpy",
) -> Tuple[np.ndarray, np.ndarray]:

    # Number of steps
    N = np.int64(T_end / dt)
    # Preallocate matrices
    x_history = np.zeros((int(T_end), particle_count), dtype=np.float64)
    v_history = np.zeros_like(x_history)
    x, v = set_initial_conditions(
        initial_dist_x=initial_dist_x,
        initial_dist_v=initial_dist_v,
        particle_count=particle_count,
        L=L,
    )

    if option.lower() == "numpy":
        step = numpy_step
    elif option.lower() == "numba":
        step = numba_step
        G = jit(nopython=True)(G)
        phi = jit(nopython=True)(phi)
    else:
        raise ValueError("Option must be numpy or numba")

    for n in range(N):
        x, v = next(step(x, v, D, dt, particle_count, L, G, phi))
        if n % (1 // dt) == 0:
            t = int(n * dt)
            x_history[t, :] = x
            v_history[t, :] = v
    return x_history, v_history


def numpy_step(
    x: np.ndarray,
    v: np.ndarray,
    D: float,
    dt: float,
    particle_count: int,
    L: float,
    G: Callable[[np.ndarray], np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Yields updated positions and velocites after one step using the Euler-Maruyama
    scheme to discretise the SDE.
    """
    while 1:
        interaction = calculate_local_interaction(x, v, phi, L)
        x = (x + v * dt) % L  # Restrict to torus
        v = (
            v
            - (v * dt)
            + G(interaction) * dt
            + np.sqrt(2 * D * dt) * np.random.normal(size=particle_count)
        )
        yield x, v


@jit(nopython=True, parallel=True)
def numba_step(
    x: np.ndarray,
    v: np.ndarray,
    D: float,
    dt: float,
    particle_count: int,
    L: float,
    G: Callable[[np.ndarray], np.ndarray],
    phi: Callable[[np.ndarray], np.ndarray],
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    while 1:
        interaction = calculate_numba_interaction(x, v, phi, L)
        for particle in range(len(x)):
            x[particle] = (x[particle] + v[particle] * dt) % L  # Restrict to torus
            v[particle] = (
                v[particle]
                - (v[particle] * dt)
                + G(interaction)[particle] * dt
                + np.sqrt(2 * D * dt) * np.random.normal()
            )
        yield x, v


def calculate_local_interaction(x: np.ndarray, v: np.ndarray, phi, L) -> np.ndarray:
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
    interaction_vector = np.zeros(len(x))
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        particle_interaction = phi(np.minimum(distance, L - distance))
        weighted_avg = np.sum(v * particle_interaction) - v[particle] * phi([0])
        scaling = np.sum(particle_interaction) - phi([0]) + 10 ** -15
        interaction_vector[particle] = weighted_avg / scaling
    return interaction_vector


@jit(nopython=True, parallel=True)
def calculate_numba_interaction(x, v, phi, L):
    interaction_vector = np.zeros(len(x), dtype=np.float64)
    for particle, position in enumerate(x):
        distance = np.abs(x - position)
        particle_interaction = phi(np.minimum(distance, L - distance))
        weighted_avg = np.sum(v * particle_interaction) - v[particle]
        scaling = np.sum(particle_interaction) - 1 + 10 ** -15
        interaction_vector[particle] = weighted_avg / scaling
    return interaction_vector


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # type: ignore

    dt = 0.1
    T_end = 1000
    x, v = get_trajectories(
        initial_dist_x="prog_spaced",
        initial_dist_v="pos_normal_dn",
        particle_count=100,
        T_end=T_end,
        dt=dt,
        L=2 * np.pi,
        D=0.5,
        G=smooth,
        phi=phis.uniform,
        option="numpy",
    )

    # plt.hist(x, density=True)
    # plt.hist(v, density=True)

    t = np.arange(0, T_end)
    ani = plotting.anim_torus(t, x, v, framestep=1, variance=0.5)
    plt.show()
# Next: non interacting particle simulation jitted
# Then: update phis, Gs to typing + docced, jitted interactions
#
