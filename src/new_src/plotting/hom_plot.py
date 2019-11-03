import numpy as np
import scipy.stats as stats

import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pickle
#Space Homogeneous

def static_line(x, sol, solver, ax):
    cmap = plt.get_cmap('plasma')
    ax.set_title(solver.__name__, fontsize=14)
    ax.set_xlabel(r'$x$', fontsize=14)
    ax.set_ylabel(r'$u$', fontsize=14)
    ax.set_prop_cycle(color=[cmap(k) for k in np.linspace(1, 0, 10)])
    for i in np.logspace(0, np.log10(np.size(sol, axis=0)-1), num=10):
        ax.plot(x, sol[int(i),])
    return ax

def anim_hist_moments(t, x, m1, var, mu=0, D=1, fs=1, animate=True, timeavg=False):
    fig, ax = plt.subplots(1, 2, figsize=(16,8))
    pos_ax = ax[0]
    mom_ax = ax[1]
    n_x, bins_x, patches_x = pos_ax.hist(x[-1,], bins=np.arange(x.min(), x.max(), 0.15),
                                             density=True, label='Position')

    sigma = np.sqrt(D)
    _x = np.arange(mu - 5*sigma, mu + 5*sigma, 0.01)
    stat_dist = stats.norm.pdf(_x, mu, sigma)

    pos_ax.plot(_x, stat_dist, '-.', label=r'Stationary D$^{\mathrm{n}}$')

    mean = np.zeros(len(x))
    var = np.zeros(len(x))
    for i in range(len(x)):
        if timeavg:
            mean[i] = np.mean(x[:i].flatten())
            var[i] = np.var(x[:i].flatten())
        else:
            mean[i] = np.mean(x[i])
            var[i] = np.var(x[i])
    line_mean, = mom_ax.plot(t, mean, label="Mean")
    line_var, = mom_ax.plot(t, var, label="Variance")
    mom_ax.legend(loc='upper right')
    mom_ax.set_xlabel('Time', fontsize=20)
    mom_ax.plot([0, t[-1]], [mu, mu], 'b-.')
    mom_ax.plot([0, t[-1]], [D, D],'r:')

    pos_ax.set_ylim(0, max(stat_dist.max(), n_x.max())+0.1)
    pos_ax.set_xlim(x.min(), x.max())
    pos_ax.set_xlabel('Position', fontsize=20)
    pos_ax.set_ylabel('Density', fontsize=20)
    ani = ax
    if animate:
        def animate(i,fs):
            line_mean.set_data(t[:(i*fs)],mean[:(i*fs)])
            line_var.set_data(t[:(i*fs)], var[:(i*fs)])
            if timeavg:
                n_x, _ = np.histogram(x[:i*fs,].flatten(),
                                  bins=np.arange(x.min(), x.max(), 0.15),
                                  density=True)
            else:
                n_x, _ = np.histogram(x[i*fs,],
                                  bins=np.arange(x.min(), x.max(), 0.15),
                                  density=True)
            if i==0:
                pos_ax.set_ylim(0, max(stat_dist.max(), n_x.max())+0.1)
            for rect_x, height_x in zip(patches_x, n_x):
                  rect_x.set_height(height_x)
            fig.suptitle('t = {:.2f}'.format(t[i*fs]), fontsize=20)
            fig.show()

        ani = animation.FuncAnimation(fig, lambda i: animate(i, fs), interval=60,
                                      frames=len(t)//fs, repeat=False)
    return ani

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
