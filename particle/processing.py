"""Run many parameter sets, pickle and save to yaml file

"""
from coolname import generate_slug
from datetime import datetime
import itertools
import numpy as np
import pandas as pd
import pathlib
import pickle
import seaborn as sns
import yaml

from particle.simulate import ParticleSystem
import particle.processing as processing


sns.set()
sns.color_palette("colorblind")

"""
Running and Saving
"""


def get_master_yaml(file_path: str = None, filename: str = "history") -> dict:
    """Get yaml from file_path
    Args:
        file_path: string containing path to history
        filename: string containing name of yaml file (default "history")

    Returns:
        dict: experiment names as keys, parameter sets as values
    """
    if file_path is None:
        file_path = ""

    try:
        with open(file_path + filename + ".yaml", "r") as file:
            history = yaml.safe_load(file)
    except Exception as e:
        print("Error reading the config file")
        return
    return history


def create_experiment_yaml(
    filename: str = "experiment", file_path: str = "Experiments/"
) -> dict:
    """Create a yaml file

        Args:
            file_path: Path to file ending in "/"

        Returns:
            dict: empty dictionary ready to write parameters to

    """
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    with open(file_path + filename + ".yaml", "w") as file:
        file.write("{}")
    with open(file_path + filename + ".yaml", "r") as file:
        experiment_yaml = yaml.safe_load(file)

    return experiment_yaml


def run_experiment(
    test_parameters: dict, history: dict = None, experiment_name: str = None
) -> None:
    """
    Take set of parameters and run simulation for all combinations in dictionary.
    """
    if history is None:
        history = processing.get_master_yaml()
    if experiment_name is None:
        experiment_name = "Experiment_" + datetime.now().strftime("%H%M-%d%m")

    exp_yaml = create_experiment_yaml(filename=experiment_name)
    history.update({experiment_name: test_parameters})
    #
    # defaults = {
    #     "particles": 100,
    #     "D": 1,
    #     "initial_dist_x": None,
    #     "initial_dist_v": None,
    #     "interaction_function": "Gamma",
    #     "dt": 0.01,
    #     "T_end": 50,
    #     "herding_function": "Step",
    #     "length": 2 * np.pi,
    #     "denominator": "Full",
    #     "well_depth": None,
    #     "gamma": 1 / 10,
    # }
    keys = list(test_parameters)
    print(history)
    # Dump test parameter superset into master file
    with open("experiments_ran.yaml", "w") as file:
        yaml.dump(history, file)

    begin = datetime.now()
    for values in itertools.product(*map(test_parameters.get, keys)):

        kwargs = dict(zip(keys, values))
        # Pickle data, generate filename and store in yaml
        filename = generate_slug(4)
        with open("Experiments/" + experiment_name + ".yaml", "w") as file:
            exp_yaml.update({filename: kwargs})
            print(exp_yaml)
            yaml.dump(exp_yaml, file)

        # Pad parameters with defaults if any missing  -- keeps yaml complete.
        # kwargs.update({k: defaults[k] for k in set(defaults) - set(kwargs)})

        # Run simulation
        print("\n Using parameters:\n")
        for parameter_name, parameter_value in kwargs.items():
            print("\t{}:  {}".format(parameter_name, parameter_value))

        start_time = datetime.now()
        x, v = ParticleSystem(**kwargs).get_trajectories()

        print("Time to solve was  {} seconds".format(datetime.now() - start_time))
        position_df = pd.DataFrame(x)
        velocity_df = pd.DataFrame(v)
        position_df.columns = position_df.columns.map(str)
        velocity_df.columns = velocity_df.columns.map(str)

        velocity_df.to_feather("Experiments/Data/" + filename + "_v")
        position_df.to_feather("Experiments/Data/" + filename + "_x")

        print("Saved at {}\n".format("Experiments/Data/" + filename))

    print("TOTAL TIME TAKEN: {}".format(datetime.now() - begin))


"""
Loading from file
"""


def get_filename(parameters: dict, history: dict) -> str:
    """
    Search yaml for matching parameter set and return filename
    """
    filename = None
    for name in history.keys():
        # print(name)
        if parameters.items() == history[name].items():
            print("Given parameters are exact match of existing set:")
            filename = name
        elif parameters.items() <= history[name].items():
            print(
                "Given parameters are subset of existing set, additional parameters are:"
            )
            additional_parameters = {
                k: history[name][k] for k in set(history[name]) ^ set(parameters)
            }
            print(additional_parameters)
            filename = name

    if filename is None:
        print("Could not find matching parameter set")
        return
    return filename


def match_parameters(fixed_parameters: dict, history: dict) -> list:
    """
    Search yaml for simulations that match fixed_parameters given
    """
    matching_files = []
    matching_files.append(get_filename(fixed_parameters, history))
    if matching_files[0] is None:
        raise ValueError("No matching parameters were found")
    print("Found {} files matching parameters".format(len(matching_files)))
    return matching_files


def load_file(
    filename: str, history: dict, file_path: str = "Simulations/"
) -> np.ndarray:
    """Get trajectory data from file

    Args:
        filename: str of 3-word random filename (obtained using get_filename)
        history: dictionary obtained from yaml file
        file_path: str of path to data if not stored in ./Simulations/

    Returns:
        np.ndarray: Position trajectory data
        np.ndarray: Velocity trajectory data

    See Also:
        :py:function:`~particle.processing.get_filename`
    """

    try:
        test_data = pickle.load(open(file_path + filename, "rb"))
    except FileNotFoundError as e:
        print("Could not find file {}".format(filename))
        return
    parameters = history.get(filename)

    x, v = test_data["Position"], test_data["Velocity"]
    t = np.arange(0, len(x) * parameters["dt"], parameters["dt"])
    return t, x, v


if __name__ == "__main__":
    file_path = "../Experiments/Simulations/"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    history = get_master_yaml("../Experiments/")
    parameters = {"T_end": [20]}
    run_experiment(parameters, history)
