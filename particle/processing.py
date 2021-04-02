"""Run many parameter sets, convert to parquet and save to yaml file

"""
from coolname import generate_slug
from datetime import datetime
import itertools
import pandas as pd
import pathlib
import warnings
import yaml

from particle.simulate import get_trajectories

"""
Running and Saving
"""


def get_main_yaml(yaml_path: str = None) -> dict:
    """Get yaml from file_path
    Args:
        file_path: string containing path to history
        filename: string containing name of yaml file (default "history")

    Returns:
        dict: experiment names as keys, parameter sets as values
    """
    if yaml_path is None:
        print("Creating new main yaml file")
        yaml_path = "experiments_ran"
        with open(yaml_path + ".yaml", "w") as file:
            file.write("{}")

    try:
        with open(yaml_path + ".yaml", "r") as file:
            history = yaml.safe_load(file)
    except FileNotFoundError:
        print("Error reading the config file")
        # raise
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
    if not pathlib.Path(file_path + filename + ".yaml").is_file():
        with open(file_path + filename + ".yaml", "w") as file:
            file.write("{}")

    with open(file_path + filename + ".yaml", "r") as file:
        experiment_yaml = yaml.safe_load(file)

    return experiment_yaml


def run_experiment(
    test_parameters: dict, history: dict = None, experiment_name: str = None,
) -> None:
    """
    Take set of parameters and run simulation for all combinations in dictionary.
    """
    if history is None:
        history = get_main_yaml()
    if experiment_name is None:
        experiment_name = "Experiment_" + datetime.now().strftime("%H%M-%d%m")

    exp_yaml = create_experiment_yaml(filename=experiment_name)
    history.update({experiment_name: test_parameters})

    keys = list(test_parameters)
    # Dump test parameter superset into main file
    with open("experiments_ran.yaml", "w") as file:
        yaml.dump(history, file)
    begin = datetime.now()
    for values in itertools.product(*map(test_parameters.get, keys)):

        kwargs = dict(zip(keys, values))

        if "dt" not in kwargs:
            kwargs["dt"] = round(kwargs["D"] * 0.1, 5)
            print(f"No timestep given, scaling dt with diffusion dt = {kwargs['dt']}")

        print("\n Using parameters:\n")
        for parameter_name, parameter_value in kwargs.items():
            print(f"\t{parameter_name}:  {parameter_value}")

        start_time = datetime.now()
        # Run simulation
        t, x, v = get_trajectories(**kwargs)

        print(f"Time to solve was  {datetime.now() - start_time} seconds")

        # Convert to Pandas df for easy conversion to feather
        time_df = pd.DataFrame(t)
        position_df = pd.DataFrame(x)
        velocity_df = pd.DataFrame(v)
        time_df.columns = time_df.columns.map(str)
        position_df.columns = position_df.columns.map(str)
        velocity_df.columns = velocity_df.columns.map(str)

        # Store as feather
        filename = generate_slug(4)
        # May need changing to TimestepExperiments/Data.nosync"
        pathlib.Path("Experiments/Data.nosync/").mkdir(parents=True, exist_ok=True)
        time_df.to_parquet(f"Experiments/Data.nosync/{filename}_t.parquet")
        velocity_df.to_parquet(f"Experiments/Data.nosync/{filename}_v.parquet")
        position_df.to_parquet(f"Experiments/Data.nosync/{filename}_x.parquet")

        with open(f"Experiments/{experiment_name}.yaml", "w") as file:
            exp_yaml.update({filename: kwargs})
            yaml.dump(exp_yaml, file)
        print(f"Saved at Experiments/Data.nosync/{filename}\n")

    print(f"TOTAL TIME TAKEN: {datetime.now() - begin}")


"""
Loading from file
"""


def match_parameters(fixed_parameters: dict, history: dict, **kwargs) -> list:
    """
    Search yaml for simulations that match fixed_parameters given
    """
    matching_files = []
    exclude = kwargs.get("exclude", {})
    for name in history.keys():
        if fixed_parameters.items() == history[name].items():
            print("Given parameters are exact match of existing set.")
            if exclude.items() <= history[name].items():
                continue
            else:
                matching_files.append(name)
        elif fixed_parameters.items() <= history[name].items():
            # print(
            #     "Given parameters are subset of existing set, additional parameters are:"
            # )
            additional_parameters = {
                k: history[name][k] for k in set(history[name]) ^ set(fixed_parameters)
            }
            if exclude and any(
                v[i] == history[name][k]
                for k, v in exclude.items()
                for i in range(len(v))
            ):
                print("Excluding...")
                continue
            else:
                matching_files.append(name)
                # print(additional_parameters)

    if not matching_files:
        warnings.warn("No matching parameters were found")

    print(f"Found {len(matching_files)} files matching parameters")
    return matching_files


def load_traj_data(
    file_name: str, data_path: str = "Experiments/Data.nosync/",
):
    """Get trajectory data from file

    Args:
        file_name: str of 3-word random filename (obtained using get_matching_files)
        history: dictionary obtained from yaml file
        data_path: str of path to data

    Returns:
        np.ndarray: Position trajectory data
        np.ndarray: Velocity trajectory data
        np.ndarray: Time data
    """
    try:
        x = pd.read_parquet(data_path + file_name + "_x.parquet").to_numpy()
        v = pd.read_parquet(data_path + file_name + "_v.parquet").to_numpy()
    except FileNotFoundError:
        print(data_path + file_name)
        print(f"Could not load file {file_name}")
        raise
    else:
        try:
            t = pd.read_parquet(data_path + file_name + "_t.parquet").to_numpy()
            return t, x, v
        except FileNotFoundError:
            print(f"No time data found for {file_name}")
            return None, x, v


def get_parameter_range(parameter, history, **kwargs):
    parameter_range = []
    exclude = kwargs.get("exclude", {})

    for parameter_dict in history.values():
        parameter_range.append(parameter_dict[parameter])
    unique_parameter_range = set(parameter_range)

    if parameter in exclude:
        print(f"Excluding {exclude[parameter]}")
        unique_parameter_range = unique_parameter_range ^ set(exclude[parameter])
    return sorted(list(unique_parameter_range))


if __name__ == "__main__":
    file_path = "../Experiments/Simulations/"
    pathlib.Path(file_path).mkdir(parents=True, exist_ok=True)
    history = get_main_yaml("../Experiments/")
    parameters = {"T_end": [20]}
    run_experiment(parameters, history)
