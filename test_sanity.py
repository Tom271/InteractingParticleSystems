import numpy as np
import het_particle as het
import scipy.stats as stats
import matplotlib.pyplot as plt
#INTERACTION CHECKS

L = 2*np.pi
interaction_vector = np.zeros(1000)
#Does phi_zero return interaction zero?
def test_zeros():
    x = np.random.uniform(low=0, high=2*np.pi, size=1000)
    v = np.random.uniform(low=-100, high=100, size=1000)
    assert het.calculate_interaction(x,v, het.phi_zero, L, interaction_vector).all() == 0.

#Does phi_one return same as average v?
def test_ones():
    x = np.random.uniform(low=0, high=2*np.pi, size=1000)
    v = np.random.uniform(low=-100, high=100, size=1000)
    out = het.calculate_interaction(x, v, het.phi_uniform, L,  interaction_vector)
    assert out.all() == np.mean(v).all()

#Does it match hand calc for 5 particles?
x = np.array([1, 6.2, 1, 0.5, 0.1])
v = np.array([-1, 0.1, 2, 5, 1])

# def test_hand():
#     #See pg81 in Notebook
#     hand_calc = [16/5 ]
#     assert calculate_interaction(x,v, phi_indicator)[0] == hand_calc[0]



#MODEL CHECKS
#Does it converge to N(0,D) when phi=0?

def test_OU():
    particles = 500
    t,x,v = het.run_particle_model(interaction_function="Zero",
            particles=particles,
            D=2,
            initial_dist_v=np.random.normal(loc=0, scale=np.sqrt(2), size=particles),
            T_end=30,
            )
    model_prob_x, _ = np.histogram(x[-500:-1,], bins=np.arange(x.min(), x.max(), 0.15),
                                         density=True)
    model_prob_v, _ = np.histogram(v[-500:-1,], bins=np.arange(v.min(), v.max(), 0.15),
                                             density=True)
    print(len(model_prob_v))
    true_prob_x = 1/(2*np.pi)*np.ones(len(model_prob_x))
    true_prob_v = stats.norm.pdf(np.arange(v.min(), v.max()-0.15, 0.15), loc=0, scale=np.sqrt(2))
    print("KL Divergence of velocity distribution:",     stats.entropy(model_prob_v, true_prob_v))
    print("L2 discrepancy of space distribution:", CL2(x[-500:,].flatten()), ", expected is ", (1/particles * (5/4 - 13/12)))


#Does it converge to N(\pm \xi) when phi=1?
def test_normal():
    particles = 500
    t,x,v = het.run_particle_model(interaction_function="Uniform",
            particles=particles,
            D=2,
            T_end=30,
            )
    model_prob_x, _ = np.histogram(x[-500:,], bins=np.arange(x.min(), x.max(), 0.15),
                                         density=True)

    model_prob_v, _ = np.histogram(v[-500:,], bins=np.arange(v.min(), v.max(), 0.15),
                                             density=True)
    true_prob_x = 1/(2*np.pi)*np.ones(len(model_prob_x))
    true_prob_v = stats.norm.pdf(np.arange(v.min(), v.max()-0.15, 0.15), loc=1, scale=np.sqrt(2))
    print("KL Divergence of velocity distribution:", stats.entropy(model_prob_v, true_prob_v))
    print("L2 discrepancy of space distribution:", CL2(x[-500:,].flatten()), ", expected is ",(1/particles * (5/4 - 13/12)))

def CL2(x, L=(2*np.pi)):
    '''Centered L2 discrepancy
    Adapted from https://stackoverflow.com/questions/50364048/
    python-removing-multiple-for-loops-for-faster-calculation-centered-l2-discrepa
    '''
    n  = len(x)

    term3 = 0
    term2 = np.sum(2. + np.abs(x/L - 0.5) - np.abs(x/L - 0.5)**2)
    for i in range(n):
        term3 += np.sum(1. + np.abs(x[i]/L - 0.5)/2 + np.abs(x/L - 0.5)/2 - np.abs(x[i]/L - x/L)/2)
    CL2 = (13/12) - (term2 - term3/n)/n

    return CL2
def test_CL2():
    N = 500

    trials = 500
    data = np.zeros(trials)
    L = 10
    for i in range(trials):
        x = np.random.uniform(low=0, high=L, size=(N,1))
        data[i] = CL2(x, L)
    fig, ax = plt.subplots(2,1)
    ax[0].plot(np.arange(0,trials), data)
    ax[0].plot([0, trials], [(1/N * (5/4 - 13/12)), 1/N * (5/4 - 13/12)])

    ax[1].hist(data, density=True)
    print(np.mean(data), np.var(data))
    print("Expected:", (1/N * (5/4 - 13/12)), 1/N**2)
    plt.show()
    assert(np.isclose(np.mean(data), (1/N * (5/4 - 13/12)), atol=1e-4))
    assert(np.isclose(np.var(data), 1/N**2, atol=1e-4))

if __name__=="__main__":
    test_zeros()
    test_ones()
    test_normal()
    test_OU()
