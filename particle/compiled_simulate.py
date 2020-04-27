"""
Using jitclass on ParticleSystem()
"""

import numpy as np
from numba import float64, int64  # import the types
from numba import jitclass

# from particle.plotting import anim_torus

# Type everything!
spec = [
    ("particles", int64),  # a simple scalar field
    ("L", float64),
    ("T_end", float64),
    ("dt", float64),
    ("D", float64),
    ("v0", float64[:]),  # a simple array field
    ("x0", float64[:]),
    ("x", float64[:, :]),
    ("v", float64[:, :]),
    ("N", int64),
    ("gamma", float64),
]


@jitclass(spec)
class ParticleSystem:
    def __init__(
        self,
        particles=200,
        L=2 * np.pi,
        T_end=10,
        dt=0.01,
        D=0.5,
        x0=np.ones(200, dtype=np.float64),
        v0=np.ones(200, dtype=np.float64),
    ):
        self.particles = particles
        self.L = L
        self.T_end = T_end
        self.dt = dt
        self.D = D

        self.v0 = v0
        self.x0 = x0

        self.N = np.int64(self.T_end / self.dt)
        self.x = np.zeros((self.N + 1, self.particles), dtype=np.float64)
        self.v = np.zeros_like(self.x)

    def calculate_interaction(self, x_curr, v_curr, gamma):
        interaction_vector = np.zeros(len(x_curr), dtype=np.float64)
        for particle, position in enumerate(x_curr):
            distance = np.abs(x_curr - position)
            particle_interaction = np.less(
                np.minimum(distance, self.L - distance), gamma * self.L
            )
            weighted_avg = np.sum(v_curr * particle_interaction) - v_curr[particle]
            scaling = np.sum(particle_interaction) - 1 + 10 ** -15
            interaction_vector[particle] = weighted_avg / scaling
        return interaction_vector

    def get_trajectories(self):
        self.x[0] = self.x0
        self.v[0] = self.v0
        gamma = 0.1
        for n in range(self.N):
            interaction = self.calculate_interaction(self.x[n], self.v[n], gamma)
            interaction = np.arctan(interaction)
            interaction /= np.arctan(1.0)
            for point in range(len(self.x[n, :])):
                self.x[n + 1, point] = (
                    self.x[n, point] + self.v[n, point] * self.dt
                ) % self.L  # Restrict to torus
                self.v[n + 1, point] = (
                    self.v[n, point]
                    - (self.v[n, point] * self.dt)
                    + interaction[point] * self.dt
                    + np.sqrt(2 * self.D * self.dt) * np.random.normal()
                )
        return self.x, self.v


if __name__ == "__main__":
    # import matplotlib.pyplot as plt
    from datetime import datetime

    startTime = datetime.now()
    particles = 100
    x0 = np.float64(np.random.default_rng().normal(loc=0.5, scale=3, size=particles))
    v0 = np.float64(np.random.default_rng().normal(loc=0.5, scale=3, size=particles))
    PS = ParticleSystem(x0=x0, v0=v0, particles=particles, T_end=50)
    x, v = PS.get_trajectories()
    dt = 0.01
    t = np.arange(0, len(x[:, 0]) + dt, dt)
    print(f"Time Taken, jitted : {datetime.now()- startTime}")
    # anim = anim_torus(t, x, v, variance=0.5, framestep=10)
    # plt.show()
    print(PS.particles)
    print(PS.L)
    print(PS.T_end)
    print(PS.dt)

    print(PS.D)
