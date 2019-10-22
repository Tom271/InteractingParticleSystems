import numpy as np
import scipy.stats as stats

import seaborn as sns
sns.set()
sns.color_palette('colorblind')

import matplotlib.pyplot as plt
import matplotlib.animation as animation

import pickle

def plot_torus(time_point, t, x, v, ax):
    '''return animated torus to be plotted on grid '''
    #fig, ax = plt.subplots()
    an = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(an), np.sin(an),'--', alpha = 0.5)
    ax.axis('equal')
    #fig.suptitle('t = {}'.format(t[0]), fontsize = 25)

    ### Plotting particles on torus #####
    ax.set_ylim(-1.1 ,1.1)
    ax.set_xlim(-1.1 ,1.1)

    #Horrible trick to plot different colours -- look for way to pass color argument in one go
    pos_vel = [x[time_point, idx] if vel >= 0 else None for  idx, vel in enumerate(v[time_point,:])]
    pos_vel = [x for x in pos_vel if x is not None]
    neg_vel = [x[time_point, idx] if vel <= 0 else None for  idx, vel in enumerate(v[time_point,:])]
    neg_vel = [x for x in neg_vel if x is not None]
    #x and y wrong way round so that +ve vel is clockwise
    neg_points, = ax.plot(np.sin(neg_vel), np.cos(neg_vel), linestyle='None',
                          marker='o')
    pos_points, = ax.plot(np.sin(pos_vel), np.cos(pos_vel), linestyle='None',
                          marker='o')

    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('x', fontsize = 25)
    ax.set_ylabel('y', fontsize = 25)

    return ax


def plot_vel_hist(time_point, t, x, v, ax):
    '''return final histogram '''
    n, bins, patches = ax.hist(v[time_point,].flatten(),  bins=np.arange(v.min(), v.max(), 0.15),
                               density=True, label='Velocity')

    ax.set_ylim(0,1.05)
    ax.set_xlim(v.min(), v.max())

    mu = np.sign(np.mean(v[0,:]))
    sigma = 1

    _v = np.arange(mu - 5*sigma, mu + 5*sigma, 0.01)
    ax.plot(_v, stats.norm.pdf(_v, mu, sigma), label=r'Stationary D$^{\mathrm{n}}$')

    ax.set_xlabel('Velocity', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)
    return ax


def plot_pos_hist(time_point, t, x, v, ax):
    '''return final histogram '''
    n, bins, patches = ax.hist(x[time_point,:],  bins=np.arange(x.min(), x.max(), 0.15),
                                         density=True, label='Position')

    ax.set_ylim(0,1.05)
    ax.set_xlim(x.min(), x.max())

    def _format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_format_func))

    mu = 1/(2*np.pi)

    _x = [x.min(), x.max()]
    ax.plot(_x, [mu,mu], label = r'Stationary D$^{\mathrm{n}}$')

    ax.set_xlabel(r'Position ($\theta$)', fontsize=20)
    ax.set_ylabel('Density', fontsize=20)

    return ax

def plot_together(time_point, t, x, v):
    '''Take axes objects and plot them on a grid'''
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_alpha(0.0)
    grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)

    big_ax = plt.subplot(grid[0:2,0:2])
    pos_ax = plt.subplot(grid[0,2])
    vel_ax = plt.subplot(grid[1,2])

    plot_pos_hist(time_point, t, x, v, pos_ax)
    plot_vel_hist(time_point, t, x, v, vel_ax)
    plot_torus(time_point, t, x, v, big_ax)

    fig.show()
    
def anim_full(t, x, v, framestep=1):
    fig = plt.figure(figsize=(20,10))
    fig.patch.set_alpha(0.0)

    grid = plt.GridSpec(2, 3, wspace=0.5, hspace=0.5)

    big_ax = plt.subplot(grid[0:2,0:2])
    pos_ax = plt.subplot(grid[0,2])
    vel_ax = plt.subplot(grid[1,2])

    an = np.linspace(0, 2*np.pi, 100)
    big_ax.plot(np.cos(an), np.sin(an),'--', alpha=0.5)
    big_ax.axis('equal')
    fig.suptitle('t = {}'.format(t[0]), fontsize = 25)
    ### Plotting particles on torus #####
    big_ax.set_ylim(-1.1 ,1.1)
    big_ax.set_xlim(-1.1 ,1.1)

     #Horrible trick to plot different colours -- look for way to pass color argument in one go
    pos_vel = [x[0, idx] if vel >= 0 else None for  idx, vel in enumerate(v[0,:])]
    pos_vel = [x for x in pos_vel if x is not None]
    neg_vel = [x[0, idx] if vel <= 0 else None for  idx, vel in enumerate(v[0,:])]
    neg_vel = [x for x in neg_vel if x is not None]
    #x and y wrong way round so that +ve vel is clockwise
    neg_points, = big_ax.plot(np.sin(neg_vel), np.cos(neg_vel), linestyle='None', marker='o', alpha=0.8, ms=10)
    pos_points, = big_ax.plot(np.sin(pos_vel), np.cos(pos_vel), linestyle='None', marker='o', alpha=0.8, ms=8)

    big_ax.tick_params(axis='both', which='major', labelsize=20)
    big_ax.set_xlabel('x', fontsize=25)
    big_ax.set_ylabel('y', fontsize=25)
    #########################################

    ##Plotting vel histogram
    n_v, bins_v, patches_v = vel_ax.hist(v[0,:], bins=np.arange(v.min(), v.max(), 0.15),
                                         density=True, label='Velocity')

    vel_ax.set_ylim(0, 1.05)
    vel_ax.set_xlim(v.min(), v.max())

    mu = np.sign(np.mean(v[0,:]))
    sigma = np.sqrt(1)

    _v = np.arange(mu - 5*sigma, mu + 5*sigma, 0.01)
    vel_ax.plot(_v, stats.norm.pdf(_v, mu, sigma), label=r'Stationary D$^{\mathrm{n}}$')

    vel_ax.set_xlabel('Velocity', fontsize=20)
    vel_ax.set_ylabel('Density', fontsize=20)


    #PLotting pos histogram
    n_x, bins_x, patches_x = pos_ax.hist(x[0,:], bins=np.arange(x.min(), x.max(), 0.15),
                                         density=True, label='Position')

    pos_ax.set_ylim(0,1.05)
    pos_ax.set_xlim(x.min(), x.max())

    def format_func(value, tick_number):
        # find number of multiples of pi/2
        N = int(np.round(2 * value / np.pi))
        if N == 0:
            return "0"
        elif N == 1:
            return r"$\pi/2$"
        elif N == 2:
            return r"$\pi$"
        elif N % 2 > 0:
            return r"${0}\pi/2$".format(N)
        else:
            return r"${0}\pi$".format(N // 2)

    pos_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
    pos_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
    pos_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

    mu = 1/(2*np.pi)

    _x = [x.min(), x.max()]
    pos_ax.plot(_x, [mu,mu], label=r'Stationary D$^{\mathrm{n}}$')

    pos_ax.set_xlabel(r'Position ($\theta$)', fontsize=20)
    pos_ax.set_ylabel('Density', fontsize=20)

    def animate(i,framestep):
        #Particles
        pos_vel = [x[i*framestep,idx] if vel >= 0 else None for  idx, vel in enumerate(v[i*framestep,])]
        pos_vel = [x for x in pos_vel if x is not None]
        neg_vel = [x[i*framestep,idx] if vel <= 0 else None for  idx, vel in enumerate(v[i*framestep,])]
        neg_vel = [x for x in neg_vel if x is not None]
        pos_points.set_data(np.sin(pos_vel), np.cos(pos_vel))
        neg_points.set_data(np.sin(neg_vel), np.cos(neg_vel))
        ####

        n_v, _ = np.histogram(v[i*framestep, :],  bins=np.arange(v.min(), v.max(), 0.15), density=True)
        n_x, _ = np.histogram(x[i*framestep, :],  bins=np.arange(x.min(), x.max(), 0.15), density=True)

        #Update vel data
        for rect_v, height_v in zip(patches_v, n_v):
              rect_v.set_height(height_v)
        #Update pos data
        for rect_x, height_x in zip(patches_x, n_x):
              rect_x.set_height(height_x)

        fig.suptitle('t = {:.2f}'.format(t[i*framestep]), fontsize=25)
        fig.show()

    ani = animation.FuncAnimation(fig, lambda i: animate(i, framestep), interval=60, frames=len(t)//framestep)

    return ani
