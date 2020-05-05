import numpy as np  # type: ignore
from typing import Dict, Tuple, Union
import warnings


def set_initial_conditions(
    *,
    initial_dist_x: Union[str, np.ndarray] = None,
    initial_dist_v: Union[str, np.ndarray] = None,
    particle_count: int = 50,
    length: float = 2 * np.pi,
    **kwargs,
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
    assert length > 0 and np.issubdtype(
        type(length), np.number
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
        x0 = np.random.uniform(low=0, high=length, size=particle_count)

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
    particle_count: int, length: float = 2 * np.pi
) -> Dict[str, np.ndarray]:
    """Builds dictionary of possible initial position conditions

        Args:
            particle_count: number of particles to simulate.
            length: length of the domain.

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
    right_cluster = left_cluster + length / 2

    prog_spaced = np.array([0.5 * (n + 1) * (n + 2) for n in range(particle_count)])
    prog_spaced /= prog_spaced[-1]
    prog_spaced *= 2 * np.pi

    even_spaced = np.arange(0, length, length / particle_count)
    position_initial_conditions = {
        "uniform_dn": np.random.uniform(low=0, high=length, size=particle_count),
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
