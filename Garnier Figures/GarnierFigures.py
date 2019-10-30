import sys
sys.path.insert(1, 'C:/Users/s1415551/Documents/GitHub/InteractingParticleSystems')

import het_particle as het
import matplotlib.pyplot as plt
from numpy.random import normal
import numpy as np

def plot_figure_2():
    fig2,ax2 = plt.subplots(2,2, figsize=(12.0, 12.0))

    #Figure 2
    dt = 0.1
    particle_count = 500
    diffusion = (2**2)/2
    T_final = 100
    exp_CL2 = 1/particle_count * (5/4 - 13/12)

    # Fig 2(a-b)
    well_depth = 2
    t,x,v = het.run_particle_model(particles=particle_count,
                            dt=dt,
                            initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion), size=particle_count),
                            D=diffusion,
                            interaction_function="Garnier",
                            herding_function="Garnier",
                            T_end=T_final,
                            L=10,
                            well_depth=well_depth)

    xi = 0


    # Plot average velocity and expected
    ax2[0,0].plot(t, np.mean(v, axis=1))
    ax2[0,0].plot([0, T_final],[xi, xi],'--',c='gray')
    ax2[0,0].plot([0, T_final],[-xi, -xi],'--',c='gray')
    ax2[0,0].plot([0, T_final],[0,0],'--',c='gray')
    ax2[0,0].set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))

    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = het.CL2(x[n,], L=10)

    ax2[0,1].plot(t, CL2_vector)
    ax2[0,1].plot([0, T_final], [exp_CL2, exp_CL2])
    ax2[0,1].set(xlabel='Time', ylabel="CL2", xlim=(0,T_final), ylim=(0, 5e-3))

    # Fig 2(c-d)
    well_depth = 6
    t,x,v = het.run_particle_model(particles=particle_count,
                                dt=dt,
                                initial_dist_v=normal(loc=0, scale=np.sqrt(diffusion),size=particle_count),
                                D=diffusion,
                                interaction_function="Garnier",
                                herding_function="Garnier",
                                T_end=T_final,
                                L=10,
                                well_depth=well_depth)

    ax2[1,0].plot(t, np.mean(v, axis=1))
    xi = 5*np.sqrt((well_depth-4)/well_depth)
    ax2[1,0].plot([0, T_final],[xi, xi],'--')
    ax2[1,0].plot([0, T_final],[-xi, -xi],'--')
    ax2[1,0].plot([0, T_final],[0,0],'--')
    ax2[1,0].set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))

    for n in range(len(t)):
        CL2_vector[n] = het.CL2(x[n,], L=10)

    ax2[1,1].plot(t, CL2_vector)
    ax2[1,1].plot([0, T_final], [exp_CL2, exp_CL2], '--')
    ax2[1,1].set(xlabel='Time', ylabel="CL2", xlim=(0,T_final), ylim=(0, 3.5e-3))
    fig2.suptitle("Garnier Fig 2, Vary Well Depth")
    fig2.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig2.savefig('Figure2.jpg', format='jpg', dpi=1000)
    plt.show()


#Figure 3
def plot_figure_3():
    dt = 0.1
    particle_count = 2000
    T_final = 100
    exp_CL2 = 1/particle_count * (5/4 - 13/12)
    well_depth = 6
    xi = 5*np.sqrt((well_depth-4)/well_depth)

    fig,ax = plt.subplots(3,2, figsize=(18.0, 12.0))

    # Figure 3(a-b)
    diffusion = (0.25**2)/2
    t,x,v = het.run_particle_model(particles=particle_count,
                                dt=dt,
                                initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion),size=particle_count),
                                D=diffusion,
                                interaction_function="Garnier",
                                herding_function="Garnier",
                                T_end=T_final,
                                L=10,
                                well_depth=well_depth)
    # Plot average velocity and expected
    ax[0,0].plot(t, np.mean(v, axis=1))
    ax[0,0].plot([0, T_final],[xi, xi],'--',c='gray')
    ax[0,0].plot([0, T_final],[-xi, -xi],'--',c='gray')
    ax[0,0].plot([0, T_final],[0,0],'--',c='gray')
    ax[0,0].set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))

    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = het.CL2(x[n,], L=10)

    ax[0,1].plot(t, CL2_vector)
    ax[0,1].plot([0, T_final], [exp_CL2, exp_CL2], '--')
    ax[0,1].set(xlabel='Time', ylabel="CL2", xlim=(0,T_final))


    # Figure 3(c-d)
    diffusion = (1.**2)/2
    t,x,v = het.run_particle_model(particles=particle_count,
                                dt=dt,
                                initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion),size=particle_count),
                                D=diffusion,
                                interaction_function="Garnier",
                                herding_function="Garnier",
                                T_end=T_final,
                                L=10,
                                well_depth=well_depth)
    # Plot average velocity and expected
    ax[1,0].plot(t, np.mean(v, axis=1))
    ax[1,0].plot([0, T_final],[xi, xi],'--',c='gray')
    ax[1,0].plot([0, T_final],[-xi, -xi],'--',c='gray')
    ax[1,0].plot([0, T_final],[0,0],'--',c='gray')
    ax[1,0].set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))

    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = het.CL2(x[n,], L=10)

    ax[1,1].plot(t, CL2_vector)
    ax[1,1].plot([0, T_final], [exp_CL2, exp_CL2],'--')
    ax[1,1].set(xlabel='Time', ylabel="CL2", xlim=(0,T_final))

    # Figure 3(e-f)
    diffusion = (1.5**2)/2
    t,x,v = het.run_particle_model(particles=particle_count,
                                dt=dt,
                                initial_dist_v=normal(loc=xi, scale=np.sqrt(diffusion),size=particle_count),
                                D=diffusion,
                                interaction_function="Garnier",
                                herding_function="Garnier",
                                T_end=T_final,
                                L=10,
                                well_depth=well_depth)
    # Plot average velocity and expected
    ax[2,0].plot(t, np.mean(v, axis=1))
    ax[2,0].plot([0, T_final],[xi, xi],'--',c='gray')
    ax[2,0].plot([0, T_final],[-xi, -xi],'--',c='gray')
    ax[2,0].plot([0, T_final],[0,0],'--',c='gray')
    ax[2,0].set(xlabel='Time', ylabel="Average Velocity", xlim=(0,T_final), ylim=(-4,4))

    CL2_vector = np.zeros(len(t))
    for n in range(len(t)):
        CL2_vector[n] = het.CL2(x[n,], L=10)

    ax[2,1].plot(t, CL2_vector)
    ax[2,1].plot([0, T_final], [exp_CL2, exp_CL2], '--')
    ax[2,1].set(xlabel='Time', ylabel="CL2", xlim=(0,T_final))

    fig.suptitle("Garnier Fig 3, Vary Diffusion")
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig('Figure3.jpg', format='jpg', dpi=1000)
    plt.show()

if __name__ == "__main__":
    plot_figure_3()
