import numpy as np
import scipy.stats as stats

import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pickle
#Space Homogeneous

def animate_PDE_hist(t, v, traj, sol):
    fig, ax = plt.subplots(figsize=(20,10))
    fig.patch.set_alpha(0.0)
    ax.set_ylim(sol.min(), sol.max()+0.1)
    ax.set_xlim(v.min(), v.max())
    ax.set_xlabel('Velocity', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)

    fig.suptitle('t = {}'.format(t[0]))

    mu = np.sign(traj[0,].mean())
    sigma = 1

    #v = np.arange(mu - 5*sigma, mu + 5*sigma, 0.01)
    ax.plot(v, stats.norm.pdf(v, mu, sigma), label=r'Stationary D$^{\mathrm{n}}$')

    line, = plt.plot(v, sol[0,], label='PDE')
    ax.legend()

    ##Plotting vel histogram
    n_v, bins_v, patches_v = ax.hist(traj[0,:],  bins=np.arange(traj.min(), traj.max(), 0.15),
                                         density=True, label='Particle Model')


    def animate(i):
        line.set_ydata(sol[i,:])
        n_v, _ = np.histogram(traj[i,:],  bins=np.arange(traj.min(), traj.max(), 0.15), density=True)
        #Update vel data
        for rect_v, height_v in zip(patches_v, n_v):
              rect_v.set_height(height_v)
        fig.suptitle('t = {:.2f}'.format(t[i]), fontsize=25)
        fig.show()

    ani = animation.FuncAnimation(fig, animate, interval=60, frames=len(t))

    return ani





    if __name__ == "__main__":
        test_data = pickle.load(open('Test_Data/test_data', 'rb' ))
        t_3 = test_data['Time']
        x_3 = test_data['Position']
        v_3 = test_data['Velocity']
