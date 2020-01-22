from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

import particle.plotting as myplot
import particle.interactionfunctions as phis
import particle.herdingfunctions as Gs


class ParticleSystem:
    def __init__(
        self,
        particles=100,
        D=1,
        initial_dist_x=None,
        initial_dist_v=None,
        interaction_function="Zero",
        dt=0.1,
        T_end=100,
        herding_function="Step",
        length=2 * np.pi,
        denominator="Full",
        well_depth=None,
        gamma=1 / 10,
        stopping_time=False,
    ):
        self.particles = particles
        self.D = D
        self.initial_dist_x = initial_dist_x
        self.initial_dist_v = initial_dist_v
        self.interaction_function = interaction_function
        self.herding_function = herding_function
        self.dt = dt
        self.T_end = T_end
        self.L = length
        self.denominator = denominator
        self.gamma = gamma
        # Get interaction function from dictionary, if not valid, throw error
        interaction_functions = {
            "Garnier": lambda x: phis.Garnier(x, self.L),
            "Uniform": phis.uniform,
            "Zero": phis.zero,
            "Indicator": lambda x: phis.indicator(x, self.L),
            "Smoothed Indicator": phis.smoothed_indicator,
            "Gamma": lambda x: phis.gamma(x, self.gamma, self.L),
        }
        try:
            self.phi = interaction_functions[interaction_function]
        except KeyError as error:
            print(
                "{} is not valid. Valid interactions are {}".format(
                    error, list(interaction_functions.keys())
                )
            )
            return
        # Get herding function from dictionary, if not valid, throw error
        herding_functions = {
            "Garnier": lambda u: Gs.Garnier(u, well_depth),
            "Step": lambda u: Gs.step(u, beta=1),
            "Smooth": Gs.smooth,
            "Zero": Gs.zero,
        }

        try:
            self.G = herding_functions[herding_function]
        except KeyError as error:
            print(
                "{} is not valid. Valid herding functions are {}".format(
                    error, list(herding_functions.keys())
                )
            )
            return

    def set_inital_conditions(self):
        left_cluster = np.random.uniform(
            low=(np.pi / 2) - np.pi / 10,
            high=(np.pi / 2) + np.pi / 10,
            size=(self.particles // 2),
        )
        right_cluster = np.random.uniform(
            low=(3 * np.pi / 2) - np.pi / 10,
            high=(3 * np.pi / 2) + np.pi / 10,
            size=(self.particles // 2),
        )
        ic_xs = {
            "uniform_dn": np.random.uniform(low=0, high=self.L, size=self.particles),
            "one_cluster": np.concatenate((left_cluster, left_cluster)),
            "two_clusters": np.concatenate((left_cluster, right_cluster)),
        }
        # Hack if odd number of particles is passed
        if len(ic_xs["two_clusters"]) != self.particles:
            ic_xs["two_clusters"] = np.concatenate(
                (ic_xs["two_clusters"], np.array([0.0]))
            )
        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.x0 = ic_xs[self.initial_dist_x]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_x, (list, tuple, np.ndarray)):
                print("Using ndarray")
                self.x0 = self.initial_dist_x
            elif self.initial_dist_x is None:
                print("Using default, uniform distrbution\n")
                self.x0 = np.random.uniform(low=0, high=self.L, size=self.particles)
            else:
                print(
                    "{} is not a valid keyword. Valid initial conditions for position are {}".format(
                        error, list(ic_xs.keys())
                    )
                )
        # Initial condition in velocity
        ic_vs = {
            "pos_normal_dn": np.random.normal(
                loc=1, scale=np.sqrt(2), size=self.particles
            ),
            "neg_normal_dn": np.random.normal(
                loc=-1, scale=np.sqrt(2), size=self.particles
            ),
            "uniform_dn": np.random.uniform(low=0, high=1, size=self.particles),
            "cauchy_dn": np.random.standard_cauchy(size=self.particles),
            "gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=self.particles),
        }
        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.v0 = ic_vs[self.initial_dist_v]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_v, (list, tuple, np.ndarray)):
                print("Using ndarray for velocity distribution")
                self.v0 = self.initial_dist_v
            elif self.initial_dist_v is None:
                print("Using default, positive normal distrbution\n")
                self.v0 = np.random.normal(
                    loc=1, scale=np.sqrt(self.D), size=self.particles
                )
            else:
                print(
                    "{} is not a valid keyword. Valid initial conditions for velocity are {}".format(
                        error, list(ic_vs.keys())
                    )
                )

    def calculate_interaction(self, x_curr, v_curr):
        """Calculate interaction term of the full particle system

            Args:
                x_curr: np.array of current particle positions
                v_curr: np.array of current particle velocities
                phi: interaction function
                L: domain length, float
                denominator: string corresponding to scaling by the total number of
                    particles or the number of particles that are interacting with each
                    particle

            Returns:
                array: The calculated interaction at the current time step for each particle

            See Also:
                :py:mod:`~particle.interactionfunctions`
        """
        interaction_vector = np.zeros(len(x_curr))
        for particle, position in enumerate(x_curr):
            distance = np.abs(x_curr - position)
            particle_interaction = self.phi(np.minimum(distance, self.L - distance))
            weighted_avg = np.sum(v_curr * particle_interaction) - v_curr[
                particle
            ] * self.phi([0])
            if self.denominator == "Full":
                scaling = np.sum(particle_interaction) + 10 ** -15 - self.phi([0])
            elif self.denominator == "Garnier":
                scaling = len(x_curr)
            interaction_vector[particle] = weighted_avg / scaling
        return interaction_vector

    def EM_scheme_step(self):
        x = self.x0
        v = self.v0
        while 1:
            yield x, v
            interaction = self.calculate_interaction(x, v)
            x = (x + v * self.dt) % self.L
            v = (
                v
                + (self.G(interaction) - v) * self.dt
                + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
            )

    def get_trajectories(self, stopping_time=None):
        """ Returns n_samples from a given algorithm. """
        self.set_inital_conditions()
        trajectories = [(self.x0, self.v0)]
        step = self.EM_scheme_step()
        tau_gamma = None
        if stopping_time:
            conv_steps = 0
            five_more = iter([True, True, True, True, True, False])
            pm1 = np.sign(self.v0.mean())
            print("Running until avg vel is {}".format(np.sign(self.v0.mean())))
            while not np.isclose(
                np.mean(trajectories[-1][1]), pm1, atol=0.5e-03,
            ) or next(five_more):
                trajectories.append(next(step))
                if len(trajectories) >= self.T_end / self.dt:
                    break
            x, v = zip(*trajectories)

            t = np.arange(0, len(x) * self.dt, self.dt)
            tau_gamma = t[-1]
            print("Hitting time was {}\n".format(tau_gamma))

        else:
            t = np.arange(0, self.T_end + self.dt, self.dt)
            N = len(t) - 1
            x, v = zip(*[next(step) for _ in range(N + 1)])
        if tau_gamma is not None:
            return t, np.array(x), np.array(v), tau_gamma
        else:
            return t, np.array(x), np.array(v)


if __name__ == "__main__":

    particle_count = 1000
    diffusion = 0.5
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 500
    length = 2 * np.pi

    interaction_function = "Uniform"
    herding_function = "Step"

    initial_data_x = "uniform_dn"
    initial_data_v = "uniform_dn"
    startTime = datetime.now()

    PS = ParticleSystem(
        interaction_function=interaction_function,
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        herding_function=herding_function,
        length=length,
        well_depth=well_depth,
        denominator="Full",
        gamma=1 / 10,
    )
    t, x, v, tau = PS.get_trajectories(stopping_time=True)
    print(v.min(), v.max())
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    plt_time = datetime.now()

    ani = myplot.anim_torus(
        t,
        x,
        v,
        mu_v=1,
        variance=diffusion,
        L=length,
        framestep=1,
        vel_panel="line",
        subsample=50,
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = "indic_strong_cluster_phi_sup"
    # # annie.save(fn + ".mp4", writer="ffmpeg", fps=10)
    # print("Total time was {} seconds".format(datetime.now() - startTime))
    plt.show()
