"""Run many parameter sets, pickle and save to yaml file

"""
from coolname import generate_slug
from datetime import datetime
import itertools
import numpy as np
import pathlib
import pickle
import seaborn as sns
import yaml

from particle.simulate import ParticleSystem

sns.set()
sns.color_palette("colorblind")

"""
Running and Saving
"""


def get_yaml(file_path: str = None) -> dict:
    """Get yaml from file_path
    Note: Must be called history.yaml
    Returns:
        dict: filenames as keys, parameter sets as values
    """
    if file_path is None:
        file_path = ""

    try:
        with open(file_path + "history.yaml", "r") as file:
            history = yaml.safe_load(file)
    except Exception as e:
        print("Error reading the config file")
        return
    return history


def run_experiment(test_parameters: dict, history: dict = None):
    """
    Take set of parameters and run simulation for all combinations in dictionary.
    """
    if history is None:
        history = get_yaml()
    defaults = {
        "particles": 100,
        "D": 1,
        "initial_dist_x": None,
        "initial_dist_v": None,
        "interaction_function": "Gamma",
        "dt": 0.01,
        "T_end": 50,
        "herding_function": "Step",
        "length": 2 * np.pi,
        "denominator": "Full",
        "well_depth": None,
        "gamma": 1 / 10,
    }
    keys = list(test_parameters)
    begin = datetime.now()
    for values in itertools.product(*map(test_parameters.get, keys)):

        kwargs = dict(zip(keys, values))
        # Pad parameters with defaults if any missing  -- keeps yaml complete.
        kwargs.update({k: defaults[k] for k in set(defaults) - set(kwargs)})

        # Check if EXACT parameter set has been tested previously
        parameter_tested = False
        for name in history.keys():
            if kwargs.items() == history[name].items():
                print("Parameter set has already been ran at {}".format(name))
                parameter_tested = True
                break
        if parameter_tested:
            continue

        # Run simulation
        print("\n Using parameters:\n")
        for parameter_name, parameter_value in kwargs.items():
            print("\t{}:  {}".format(parameter_name, parameter_value))

        start_time = datetime.now()
        x, v = ParticleSystem(**kwargs).get_trajectories()

        print("Time to solve was  {} seconds".format(datetime.now() - start_time))
        test_data = {"Position": x, "Velocity": v}

        # Pickle data, generate filename and store in yaml
        filename = generate_slug(3)
        print("File name is {}".format(filename))
        pickle.dump(test_data, open(file_path + filename, "wb"))
        history.update({filename: kwargs})

        with open("history.yaml", "w") as file:
            yaml.dump(history, file)
        print("Saved at {}\n".format(file_path + filename))


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

    file_path = "Simulations/"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    history = get_yaml()
    parameters = {"T_end": 20}
    run_experiment(parameters, history)
