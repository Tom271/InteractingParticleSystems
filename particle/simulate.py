from datetime import datetime
import numpy as np

# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# import particle.plotting as myplot
import particle.interactionfunctions as phis
import particle.herdingfunctions as Gs


class ParticleSystem:
    def __init__(
        self,
        particles=100,
        D=1,
        initial_dist_x=None,
        initial_dist_v=None,
        interaction_function="Gamma",
        dt=0.01,
        T_end=50,
        herding_function="Step",
        length=2 * np.pi,
        denominator="Full",
        well_depth=None,
        gamma=1 / 10,
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
                f"{error} is not valid."
                f" Valid interactions are {list(interaction_functions.keys())}"
            )
            return
        # Get herding function from dictionary, if not valid, throw error
        herding_functions = {
            "Garnier": lambda u: Gs.Garnier(u, well_depth),
            "Hyperbola": Gs.hyperbola,
            "Smooth": Gs.smooth,
            "Step": lambda u: Gs.step(u, beta=1),
            "Symmetric": Gs.symmetric,
            "Zero": Gs.zero,
        }

        try:
            self.G = herding_functions[herding_function]
        except KeyError as error:
            print(
                f"{error} is not valid."
                f" Valid herding functions are {list(herding_functions.keys())}"
            )
            return

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
            if self.denominator == "Local":
                scaling = np.sum(particle_interaction) - self.phi([0]) + 10 ** -15
            elif self.denominator == "Global":
                scaling = len(x_curr) - 1 + 10 ** -15
            interaction_vector[particle] = weighted_avg / scaling
        return interaction_vector

    def EM_scheme_step(self):
        """
        Yields updated positions and velocites after one step using the Euler-Maruyama
        scheme to discretise the SDE.
        """
        x = self.x0
        v = self.v0
        self.interaction_data = []
        while 1:
            yield x, v
            interaction = self.calculate_interaction(x, v)
            # self.interaction_data.append(interaction)
            x = (x + v * self.dt) % self.L
            v = (
                v
                + (self.G(interaction) - v) * self.dt
                + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
            )

    def get_trajectories(self):
        """ Returns samples from a given algorithm. """
        self.set_position_initial_condition()
        self.set_velocity_initial_condition()
        step = self.EM_scheme_step()
        t = np.arange(0, self.T_end + self.dt, self.dt)
        N = len(t) - 1
        x, v = zip(*[next(step) for _ in range(N)])

        return np.array(x), np.array(v)
        # return np.array(x), np.array(v)
        # x = np.zeros((N + 1, self.particles), dtype=float)
        # v = np.zeros_like(x)
        # x[0] = self.x0
        # v[0] = self.v0
        # for n in range(N):
        #     interaction = self.calculate_interaction(x[n], v[n])
        #     x[n + 1,] = (x[n,] + v[n,] * self.dt) % self.L  # Restrict to torus
        #     v[n + 1,] = (
        #         v[n,]
        #         - (v[n,] * self.dt)
        #         + self.G(interaction) * self.dt
        #         + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
        #     )
        # return x, v

    def get_stopping_time(self):  # NOT WORKING!!
        """Returns the stopping time without storing trajectories """
        tau_gamma = 0
        self.set_position_initial_condition()
        self.set_velocity_initial_condition()

        x, v = self.x0, self.v0
        conv_steps = [True for _ in range(int(1 / self.dt))]
        conv_steps.append(False)
        n_more = iter(conv_steps)
        step = self.EM_scheme_step()
        pm1 = np.sign(self.v0.mean())  # What if ==0?
        print(pm1)
        print(f"Running until avg vel is {np.sign(self.v0.mean())}")
        while not np.isclose(v.mean(), pm1, atol=0.5e-2) or next(n_more):
            x, v = next(step)
            tau_gamma += self.dt
            if tau_gamma >= self.T_end:
                print("Did not converge to pm1")
                break

        if np.isclose(v.mean(), 0, atol=0.1e-2):
            print("Hit 0")
            tau_gamma = 10 ** 10
        print(f"Hitting time was {tau_gamma}\n")

        return tau_gamma

    def set_position_initial_condition(self):
        def _cluster(particles: int, loc: float, width: float) -> np.ndarray:
            cluster = np.random.uniform(
                low=loc - width / 2, high=loc + width / 2, size=particles
            )
            return cluster

        area_left_cluster = _cluster(
            particles=2 * self.particles // 3, loc=np.pi, width=np.pi / 5
        )
        area_right_cluster = _cluster(
            particles=self.particles // 3, loc=0, width=np.pi / 5
        )

        prog_spaced = np.array([0.5 * (n + 1) * (n + 2) for n in range(self.particles)])
        prog_spaced /= prog_spaced[-1]
        prog_spaced *= 2 * np.pi

        even_spaced = np.arange(0, 2 * np.pi, 2 * np.pi / self.particles)
        ic_xs = {
            "uniform_dn": np.random.uniform(low=0, high=self.L, size=self.particles),
            # "one_cluster": np.concatenate((left_cluster, left_cluster)),
            # "two_clusters": np.concatenate((left_cluster, right_cluster)),
            "two_clusters_2N_N": np.concatenate(
                (area_left_cluster, area_right_cluster)
            ),
            "bottom_cluster": _cluster(
                particles=self.particles, loc=np.pi, width=np.pi / 5
            ),
            "top_cluster": _cluster(particles=self.particles, loc=0.0, width=np.pi / 5),
            "even_spaced": even_spaced,
            "prog_spaced": prog_spaced,
        }

        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.x0 = ic_xs[self.initial_dist_x]
            # Hack if odd particle number
            while len(self.x0) != self.particles:
                self.x0 = np.concatenate((self.x0, [self.x0[-1]]))
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_x, (list, tuple, np.ndarray)):
                print("Using ndarray for position distribution")
                self.x0 = np.array(self.initial_dist_x)
            elif self.initial_dist_x is None:
                print("Using default, uniform distrbution for positions\n")
                self.x0 = np.random.uniform(low=0, high=self.L, size=self.particles)
            else:
                print(
                    f"{error} is not a valid keyword."
                    f" Valid initial conditions for position are { list(ic_xs.keys())}"
                ),

    def set_velocity_initial_condition(self):
        # Initial condition in velocity
        slower_pos = np.random.uniform(low=0, high=1, size=(2 * self.particles) // 3)
        faster_pos = np.random.uniform(low=1, high=2, size=(self.particles // 3))

        left_NN_cluster = -0.2 * np.ones(2 * self.particles // 3)
        right_N_cluster = 1.8 * np.ones(self.particles // 3)

        normal_left_NN_cluster = -0.2 + np.random.normal(
            scale=0.5, size=2 * self.particles // 3
        )
        normal_right_N_cluster = 1.8 + np.random.normal(
            scale=0.5, size=self.particles // 3
        )

        left_NN_cluster_0 = -0.45 * np.ones(2 * self.particles // 3)
        right_N_cluster_0 = 0.9 * np.ones(self.particles // 3)

        ic_vs = {
            "pos_normal_dn": np.random.normal(
                loc=1.2, scale=np.sqrt(2), size=self.particles
            ),
            "neg_normal_dn": np.random.normal(
                loc=-1.2, scale=np.sqrt(2), size=self.particles
            ),
            "uniform_dn": np.random.uniform(low=0, high=1, size=self.particles),
            "pos_gamma_dn": np.random.gamma(shape=7.5, scale=1.0, size=self.particles),
            "neg_gamma_dn": -np.random.gamma(shape=7.5, scale=1.0, size=self.particles),
            "pos_const_near_0": 0.2 * np.ones(self.particles),
            "neg_const_near_0": -0.2 * np.ones(self.particles),
            "pos_const": 1.8 * np.ones(self.particles),
            "neg_const": -1.8 * np.ones(self.particles),
            "2N_N_cluster_const": np.concatenate((left_NN_cluster, right_N_cluster)),
            "2N_N_cluster_normal": np.concatenate(
                (normal_left_NN_cluster, normal_right_N_cluster)
            ),
            "2N_N_cluster_avg_0": np.concatenate(
                (left_NN_cluster_0, right_N_cluster_0)
            ),
        }

        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.v0 = ic_vs[self.initial_dist_v]
            # Hack if odd particle number
            while len(self.v0) != self.particles:
                self.v0 = np.concatenate((self.v0, [self.v0[-1]]))
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_v, (list, tuple, np.ndarray)):
                print("Using ndarray for velocity distribution")
                self.v0 = np.array(self.initial_dist_v)
            elif self.initial_dist_v is None:
                print("Using default, positive normal distrbution\n")
                self.v0 = np.random.normal(
                    loc=1, scale=np.sqrt(self.D), size=self.particles
                )
            else:
                print(
                    f"{error} is not a valid keyword. Valid initial conditions for"
                    f" velocity are {list(ic_vs.keys())}"
                )


if __name__ == "__main__":

    particle_count = 100
    diffusion = 0.5
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 50
    length = 2 * np.pi

    interaction_function = "Gamma"
    herding_function = "Step"
    initial_data_x = "uniform_dn"
    initial_data_v = "pos_normal_dn"
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
        denominator="Local",
        gamma=1 / 10,
    )
    x, v = PS.get_trajectories()
    t = np.arange(0, len(x) * timestep, timestep)
    # print(v.min(), v.max())
    print(f"Time to solve was  {datetime.now() - startTime} seconds")
    plt_time = datetime.now()

    # ani = myplot.anim_torus(
    #     t, x, v, mu_v=1, variance=diffusion, L=length, framestep=1, subsample=50,
    # )
    # print(f"Time to plot was  {datetime.now() - plt_time} seconds")
    # plt.show()
    # fn = "MANY_PARTICLE"
    # writer = animation.FFMpegWriter(
    #     fps=20, extra_args=["-vcodec", "libx264"], bitrate=-1
    # )
    # ani.save(fn + ".mp4", writer=writer, dpi=200)

    # print(f"Total time was {datetime.now() - startTime} seconds")
