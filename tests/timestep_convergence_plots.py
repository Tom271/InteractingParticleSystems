import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import solve_ivp
import seaborn as sns

import particle.plotting as plotting
import particle.processing as processing
import particle.statistics as statistics
import particle.herdingfunctions as G


def plot_all_average_velocity_and_variance(
    data_path: str, yaml_path: str, search_parameters: dict = {}
):
    sns.set(style="white", context="talk")
    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path)
    timesteps = processing.get_parameter_range("dt", history, exclude={"dt": 1.0})

    fig = plotting.multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=statistics.calculate_avg_vel,
        parameter="dt",
        parameter_range=timesteps,
        history=history,
        include_traj=False,
        data_path=data_path,
    )

    fig2 = plotting.multiple_timescale_plot(
        search_parameters,
        break_time_step=40,
        metric=statistics.calculate_variance,
        parameter="dt",
        parameter_range=timesteps,
        history=history,
        include_traj=False,
        data_path=data_path,
    )
    plt.show()
    return


def plot_ODE_solution(
    data_path: str, yaml_path: str, search_parameters: dict = {}, **kwargs
):
    os.chdir("E:/")
    history = processing.get_master_yaml(yaml_path)

    fig, ax = plt.subplots()
    timestep_range = processing.get_parameter_range("dt", history, **kwargs)
    for timestep in timestep_range:
        search_parameters["dt"] = timestep
        file_names = processing.match_parameters(search_parameters, history, **kwargs)
        for file_name in file_names:
            t, x, v = processing.load_traj_data(file_name, data_path)
            # print(len(t))
            if file_name == file_names[0]:
                sum_avg_vel = np.zeros(len(v[:, 0]))
            print(file_name)
            # sum_avg_vel += v.mean(axis=1)
            ax.plot(t, v.mean(axis=1), "r--", alpha=0.1)

    def first_moment_ode(t, M):
        return G.step(M) - M

    sol = solve_ivp(
        first_moment_ode,
        (t.min(), t.max()),
        [v[0].mean()],
        t_eval=t.ravel(),
        rtol=10 ** -9,
        atol=10 ** -9,
    )
    t = sol.t
    M = sol.y
    ax.plot(t, M.ravel(), label="Numerical Sol")
    ax.plot(t, sum_avg_vel / len(file_names))
    ax.legend()

    plt.show()
    return


if __name__ == "__main__":
    yaml_path = "TimestepExperiments/UniformInteractionLoweringTimestep"
    data_path = "./TimestepExperiments/Data.nosync/"
    search_parameters = {"phi": "Uniform", "particles": 99}
    plot_ODE_solution(
        data_path,
        yaml_path,
        exclude={
            "dt": [1.0, 0.5179474679231212, 0.2682695795279726, 0.037275937203149416]
        },
    )
