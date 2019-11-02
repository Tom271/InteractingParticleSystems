from garnier_space_het import *
import perfplot
import herding as herd

def foo(x):
    return 0

def test_run(N):
    particle_count = N
    diffusion = 0.5
    well_depth = 6
    timestep = 0.1
    T_final = 20

    interaction_function = phi_Garnier
    herding_function = herd.step_G#(lambda u: G_Garnier(u, well_depth))

    # Set initial data for Gaussian
    mu_init = 5*np.sqrt((well_depth-4)/well_depth)
    sd_init = np.sqrt(diffusion**2 / 2)

    # Set max/min for indicator
    max_init = 2
    min_init = 1

    gaussian = {
        "particle": normal(loc=mu_init, scale=sd_init, size=particle_count),
        "pde": lambda x: stats.norm.pdf(x, loc=mu_init, scale=sd_init),
    }

    initial_data_x = None
    initial_data_v = gaussian["particle"]  # Choose indicator or gaussian
    t,x,v = run_particle_model(
        phi=interaction_function,
        particles=particle_count,
        D=diffusion,
        initial_dist_x=initial_data_x,
        initial_dist_v=initial_data_v,
        dt=timestep,
        T_end=T_final,
        G=herding_function,
    )
    return t, x, v

# import numpy
# import perfplot
#
# perfplot.show(
#     setup=lambda n: numpy.random.randint(n),  # or simply setup=numpy.random.rand
#     kernels=[
#         lambda a: test_run(a),
#     ],
#     labels=["run_particle"],
#     n_range=[1,2,5,10,15,20,25,50,75,100,150,200,300,400,500,600,700,800,900,1000],
#     xlabel="Particle Count",
#     # More optional arguments with their default values:
#     title="How does particle count affect runtime?",
#     # logx=False,
#     logy=True,
#     equality_check=None,  # set to None to disable "correctness" assertion
#     # automatic_order=True,
#     # colors=None,
#     # target_time_per_measurement=1.0,
#     # time_unit="auto"  # set to one of ("s", "ms", "us", or "ns") to force plot units
# )
