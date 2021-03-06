import numpy as np
import pytest
import particle.interactionfunctions as phis


rng = np.random.default_rng()
particle_count = rng.integers(low=1, high=1000)
distance_data = rng.uniform(low=0, high=np.pi, size=particle_count)

interaction_functions = {
    "Garnier": phis.Garnier,
    "Uniform": phis.uniform,
    "Zero": phis.zero,
    "Indicator": phis.indicator,
    "Smoothed Indicator": phis.smoothed_indicator,
    "Gamma": phis.gamma,
    "Normalised Gamma": phis.normalised_gamma,
    "Gaussian": phis.gaussian,
    "Bump": phis.bump,
}


class TestInteractions:
    @pytest.mark.parametrize("phi_fn", interaction_functions.values())
    def test_output_length(self, phi_fn):
        np.testing.assert_(len(phi_fn(distance_data)) == len(distance_data))

    @pytest.mark.parametrize("phi_fn", interaction_functions.values())
    def test_non_negative(self, phi_fn):
        interaction = phi_fn(distance_data)
        np.testing.assert_(np.less_equal(np.zeros_like(interaction), interaction).all())


class TestGamma:
    @pytest.mark.parametrize("gamma", np.arange(0, 0.51, 0.01))
    def test_output_length(self, gamma):
        np.testing.assert_(len(phis.gamma(distance_data, gamma)) == len(distance_data))

    @pytest.mark.parametrize("gamma", np.arange(0, 0.51, 0.01))
    def test_non_negative(self, gamma):
        interaction = phis.gamma(distance_data, gamma)
        np.testing.assert_(np.less_equal(np.zeros_like(interaction), interaction).all())

    def test_gamma_is_uniform(self):
        np.testing.assert_equal(
            phis.gamma(distance_data, L=2 * np.pi, gamma=0.5),
            phis.uniform(distance_data),
        )

    def test_gamma_is_zero(self):
        np.testing.assert_equal(
            phis.gamma(distance_data, L=2 * np.pi, gamma=0), phis.zero(distance_data),
        )
