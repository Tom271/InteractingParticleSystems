import numpy as np
import pytest

from particle.simulate import calculate_local_interaction
import particle.herdingfunctions as Gs
import particle.interactionfunctions as phis

rng = np.random.default_rng()
particle_count = rng.integers(low=1, high=1000)
x = rng.uniform(low=0, high=2 * np.pi, size=particle_count)
v = rng.normal(loc=0, scale=3, size=particle_count)
interaction_data = calculate_local_interaction(x, v, phis.gamma, self_interaction=1)

herding_functions = {
    "Garnier": Gs.Garnier,
    "Hyperbola": Gs.hyperbola,
    "Smooth": Gs.smooth,
    "Alpha Smooth": lambda u: Gs.alpha_smooth(u, alpha=10),
    "Step": Gs.step,
    "Symmetric": lambda u: Gs.symmetric(u, alpha=2),
    "Zero": Gs.zero,
}


class TestHerding:
    @pytest.mark.parametrize("herding_fn", herding_functions.values())
    def test_output_length(self, herding_fn):
        np.testing.assert_(len(herding_fn(interaction_data)) == len(interaction_data))

    @pytest.mark.parametrize("herding_fn", herding_functions.values())
    def test_odd(self, herding_fn):
        np.testing.assert_equal(
            -herding_fn(interaction_data), herding_fn(-interaction_data)
        )

    @pytest.mark.parametrize("herding_fn", herding_functions.values())
    def test_zero_at_zero(self, herding_fn):
        np.testing.assert_equal(herding_fn(np.array([0.0])), [0.0])
