import particle.simulate_v2 as simulate

x, v = simulate.set_initial_conditions(
    initial_dist_x="uniform_dn", initial_dist_v="2N_N_cluster_const", particle_count=10
)


def test_length():
    assert len(x) == len(v)


""" tests for initial conditions:
 - odd particle number
 - even number
 - not multiple of three for clusters
 - passing array as argument
 - passing no argument
"""

print(x, v)
