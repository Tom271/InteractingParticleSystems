from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


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
        prog_spaced = np.array([0.5 * (n + 1) * (n + 2) for n in range(self.particles)])
        prog_spaced /= prog_spaced[-1]
        prog_spaced *= 2 * np.pi

        even_spaced = np.arange(0, 2 * np.pi, 2 * np.pi / self.particles)
        ic_xs = {
            "uniform_dn": np.random.uniform(low=0, high=self.L, size=self.particles),
            "one_cluster": np.concatenate((left_cluster, left_cluster)),
            "two_clusters": np.concatenate((left_cluster, right_cluster)),
            "even_spaced": even_spaced,
            "prog_spaced": prog_spaced,
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
                print("Using ndarray for position distribution")
                self.x0 = np.array(self.initial_dist_x)
            elif self.initial_dist_x is None:
                print("Using default, uniform distrbution for positions\n")
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
            "const": 1.8 * np.ones(self.particles),
            "-const": -1.8 * np.ones(self.particles),
        }
        # Try using dictionary to get IC, if not check if input is array, else use a
        # default IC
        try:
            self.v0 = ic_vs[self.initial_dist_v]
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
        """
        Yields updated positions and velocites after one step using the Euler-Maruyama
        scheme to discretise the SDE.
        """
        x = self.x0
        v = self.v0
        self.interaction_data = []
        while 1:
            interaction = self.calculate_interaction(x, v)
            self.interaction_data.append(interaction)
            x = (x + v * self.dt) % self.L
            v = (
                v
                + (self.G(interaction) - v) * self.dt
                + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
            )
            yield x, v

    def get_trajectories(self):
        """ Returns samples from a given algorithm. """
        self.set_inital_conditions()
        trajectories = [(self.x0, self.v0)]
        step = self.EM_scheme_step()
        t = np.arange(0, self.T_end + self.dt, self.dt)
        N = len(t) - 1
        x, v = zip(*[next(step) for _ in range(N + 1)])
        t = np.arange(0, len(trajectories[-1][0]) * self.dt + self.dt, self.dt)
        return t, np.array(x), np.array(v)

    def get_stopping_time(self):
        """Returns the stopping time without storing trajectories """
        tau_gamma = 0
        x, v = self.x0, self.v0
        conv_steps = [True for _ in range(1 / self.dt)]
        conv_steps.append(False)
        n_more = iter(conv_steps)
        step = self.EM_scheme_step()
        pm1 = np.sign(self.v0.mean())  # What if ==0?
        print("Running until avg vel is {}".format(np.sign(self.v0.mean())))
        while (
            not np.isclose(v.mean(), pm1, atol=0.5e-03)
            or not np.isclose(v.mean(), 0, atol=0.5e-3)
            or next(n_more)
        ):
            x, v = next(step)
            tau_gamma += self.dt

        if np.isclose(v.mean(), 0, atol=0.1e-2):
            print("Hit 0")
            tau_gamma = 10 ** 10
        print("Hitting time was {}\n".format(tau_gamma))

        return tau_gamma


def calculate_stopping_time(v, dt, expect_converge_value):
    """Given a velocity trajectory, calculate the time to convergence.
     """
    tol = 0.5e-2
    zero_mask = np.isclose(np.mean(v, axis=1), 0, atol=tol)
    one_mask = np.isclose(np.mean(v, axis=1), 1, atol=tol)
    neg_one_mask = np.isclose(np.mean(v, axis=1), -1, atol=tol)
    # expect_converge_value = np.sign(np.mean(v[0, :]))
    conv_steps = [True for _ in range(int(1 / dt))]
    conv_steps.append(False)
    print(expect_converge_value)
    if expect_converge_value == 1.0:
        count = 0
        n_more = iter(conv_steps)
        while not one_mask[count] or next(n_more):
            tau = count * dt
            count += 1
            if count >= len(one_mask):
                break
    elif expect_converge_value == 0.0:
        count = 0
        n_more = iter(conv_steps)
        while not zero_mask[count] or next(n_more):
            tau = count * dt
            count += 1
            if count >= len(zero_mask):
                break
    elif expect_converge_value == -1.0:
        count = 0
        n_more = iter(conv_steps)
        while not neg_one_mask[count] or next(n_more):
            tau = count * dt
            count += 1
            if count >= len(neg_one_mask):
                break
    else:
        print("expect_converge_value is", expect_converge_value)
    return tau


if __name__ == "__main__":

    particle_count = 1000
    diffusion = 0.5
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 50
    length = 2 * np.pi

    interaction_function = "Uniform"
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
        denominator="Full",
        gamma=1 / 10,
    )
    t, x, v = PS.get_trajectories()
    print(v.min(), v.max())
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    plt_time = datetime.now()

    ani = myplot.anim_torus(
        t, x, v, mu_v=1, variance=diffusion, L=length, framestep=1, subsample=50,
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    plt.show()
    fn = "MANY_PARTICLE"
    writer = animation.FFMpegWriter(
        fps=20, extra_args=["-vcodec", "libx264"], bitrate=-1
    )
    # ani.save(fn + ".mp4", writer=writer, dpi=200)

    # print("Total time was {} seconds".format(datetime.now() - startTime))
