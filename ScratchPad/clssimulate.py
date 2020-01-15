from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

# import seaborn as sns

import particle.interactionfunctions as phis
import particle.herdingfunctions as Gs
import particle.plotting as myplot

# sns.set()
# sns.color_palette("colorblind")


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
            self.x[0,] = ic_xs[self.initial_dist_x]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_x, (list, tuple, np.ndarray)):
                print("Using ndarray")
                self.x[0,] = self.initial_dist_x
            elif self.initial_dist_x is None:
                print("Using default, uniform distrbution\n")
                self.x[0,] = np.random.uniform(low=0, high=self.L, size=self.particles)
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
            self.v[0,] = ic_vs[self.initial_dist_v]
        except (KeyError, TypeError) as error:
            if isinstance(self.initial_dist_v, (list, tuple, np.ndarray)):
                print("Using ndarray for velocity distribution")
                self.v[0,] = self.initial_dist_v
            elif self.initial_dist_v is None:
                print("Using default, positive normal distrbution\n")
                self.v[0,] = np.random.normal(
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
            weighted_avg = np.sum(v_curr * particle_interaction)
            if self.denominator == "Full":
                scaling = np.sum(particle_interaction) + 10 ** -50
            elif self.denominator == "Garnier":
                scaling = len(x_curr)
            interaction_vector[particle] = weighted_avg / scaling
        return interaction_vector

    def EM_scheme_step(self, x, v):
        # x = self.x[
        #     0,
        # ]
        # v = self.v[
        #     0,
        # ]
        while 1:
            yield x, v
            interaction = self.calculate_interaction(x, v)
            print(interaction)
            x = (x + v * self.dt) % self.L
            v = (
                v
                + (self.G(interaction) - v) * self.dt
                + np.sqrt(2 * self.D * self.dt) * np.random.normal(size=self.particles)
            )
            print(x, v)

    def get_samples(self, stopping_time=None):
        """ Returns n_samples from a given algorithm. """
        t = np.arange(0, self.T_end + self.dt, self.dt)
        N = len(t) - 1
        self.x = np.zeros((N + 1, self.particles), dtype=float)
        self.v = np.zeros((N + 1, self.particles), dtype=float)
        self.set_inital_conditions()

        if stopping_time:
            return
        else:
            for n in range(N):
                interaction = self.calculate_interaction(self.x[n], self.v[n])
                self.x[n + 1,] = (
                    self.x[n,] + self.v[n,] * self.dt
                ) % self.L  # Restrict to torus
                self.v[n + 1,] = (
                    self.v[n,]
                    - (self.v[n,] * self.dt)
                    + self.G(interaction) * self.dt
                    + np.sqrt(2 * self.D * self.dt)
                    * np.random.normal(size=self.particles)
                )
            self.t = np.arange(0, self.T_end + self.dt, self.dt)
            return self.t, self.x, self.v


if __name__ == "__main__":

    particle_count = 1024
    diffusion = 0.5
    well_depth = 10
    xi = 5 * np.sqrt((well_depth - 4) / well_depth)
    timestep = 0.1
    T_final = 30
    length = 2 * np.pi

    interaction_function = "Uniform"
    herding_function = "Step"

    initial_data_x = np.ones(particle_count) - 0.5
    initial_data_v = np.ones(particle_count)  # Choose indicator or gaussian
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
        gamma=0,
    )
    t, x, v = PS.get_samples()
    print("Time to solve was  {} seconds".format(datetime.now() - startTime))
    # g = sns.jointplot(x.flatten(), v.flatten(), kind="hex", height=7, space=0)
    # g.ax_joint.set_xlabel("Position", fontsize=16)
    # g.ax_joint.set_ylabel("Velocity", fontsize=16)
    # plt.show()
    plt_time = datetime.now()
    # model_prob_x, _ = np.histogram(x[-500:-1,].flatten(), bins=np.arange(x.min(), x.max(), 0.15),
    #                                      density=True)
    # model_prob_v, _ = np.histogram(v[-500:-1,].flatten(), bins=np.arange(v.min(), v.max(), 0.15),
    #                                 density=True)
    # model_prob_x, _ = np.histogram(x[-1,], bins=np.arange(x.min(), x.max(), 0.15),
    #                                      density=True)
    # model_prob_v, _ = np.histogram(v[-1,], bins=np.arange(v.min(), v.max(), 0.15),
    #                                 density=True)
    # fig, ax = plt.subplots(1,2, figsize=(24 ,12))
    # ax[0].hist(x[-1,], bins=np.arange(x.min(), x.max(), 0.15), density=True)
    # ax[0].plot([x.min(),x.max()], [1/length ,1/length], '--')
    # ax[0].set(xlabel='Position')
    #
    # ax[1].hist(v[-1,], bins=np.arange(v.min(), v.max(), 0.15),
    #                                       density=True)
    # ax[1].plot(np.arange(-v.max(),v.max(),0.01), stats.norm.pdf(np.arange(-v.max(),v.max(),0.01), loc=xi, scale=np.sqrt(diffusion)), '--')
    # ax[1].set(xlabel='Velocity')
    # true_prob_x = 1/(2*np.pi)*np.ones(len(model_prob_x))
    # true_prob_v = stats.norm.pdf(np.arange(v.min(), v.max()-0.15, 0.15), loc=0, scale=np.sqrt(diffusion))
    # fig.savefig('smallwellxvhist.jpg', format='jpg', dpi=250)

    # print("KL Divergence of velocity distribution:",     stats.entropy(model_prob_v, true_prob_v))
    # annie = hetplt.anim_full(t, x, v, mu=xi, variance=diffusion, L=length, framestep=1)
    # annie = hetplt.anim_full(
    #     t, x, v, mu_v=xi, variance=diffusion, L=length, framestep=1
    # )
    ani = myplot.anim_torus(
        t,
        x,
        v,
        mu_v=1,
        variance=0.5,  # np.sqrt(default_parameters["D"]),
        L=length,
        framestep=1,
    )
    print("Time to plot was  {} seconds".format(datetime.now() - plt_time))
    fn = "indic_strong_cluster_phi_sup"
    # # annie.save(fn + ".mp4", writer="ffmpeg", fps=10)
    # print("Total time was {} seconds".format(datetime.now() - startTime))
    plt.show()
