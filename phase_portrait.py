#!/usr/bin/env python2

import sys as sys
import numpy as np
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt

def main():

    # Set up list of environments to run over
    r = np.array([5.])
    N = np.array([8.])
    d = np.array([0.8])
    env_list = epgg.create_environments(r, N, d)

    # Set up list of strategies to compete
    i = np.array([0.01])
    k = np.array([0.])
    n = np.array([0.])
    strats_list = epgg.create_strategies(i, k, n)

    # Set up list of initial conditions, as these are plotted in the u, q space, it is easiest to
    # give initial conditions as such, but they must be then converted to Y, X (individual species
    # frequencies.
    u0 = np.array([0.66]) 
    q0 = np.array([0.85, 0.4])
    x0_list = epgg.create_initial_frequency_list([u0, q0]) 

    u0 = np.array([0.8])
    q0 = np.array([0.6, 0.15])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([0.9])
    q0 = np.array([0.05])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    x0_list = np.array([[u*(1-q), u*q] for [u, q] in x0_list]) #transformation mentioned just above
    
    # Initialize community type
    hd = epgg.HauertDoebeli()

    # Calculate trajectories and fixed points
    trajectories = epgg.evolve_trajectories(hd, env_list, strats_list, x0_list)
    fixed_points = epgg.calculate_fixed_points(hd, env_list, strats_list)

    # Prepare figure and empty plot
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)

    plot_trajectories(plot, trajectories)
    plot_fixed_points(plot, fixed_points)

    plot_features(plot)

    fig.tight_layout()
    fig.savefig(epgg.VPATH + 'phase_plot.pdf')
    fig.clf()

def plot_trajectories(plot, trajectories, ulog=False, lwidth=4):
    """Adds trajectories to plot with appropriate colors.
    
    Trajectories are structured in a hiarchy of lists grouping them by environment, strategies and
    initial frequencies.  Thus a trajectory is retrived as 
    traj = trajectories[environment #][strategy #][initial frequencies #]. 
    
    """

    for e in np.arange(trajectories.shape[0]): 
        for s in np.arange(trajectories.shape[1]):
            for x in np.arange(trajectories.shape[2]):

                u = np.sum(trajectories[e][s][x], axis=1)
                q = trajectories[e][s][x][:, 1] / u

                if ulog:
                    u = np.log10(u) # we want to plot this on a log scale

                if ((ulog == False and u[-1] < 1e-6) or (ulog == True and u[-1] < -3) or
                        np.isnan(u[-1])):
                    plot.plot(u, q, '#a1622b', lw=lwidth)
                elif 1-q[-1] < 1e-5:
                    plot.plot(u, q, '#1B5C62', lw=lwidth)
                else:
                    plot.plot(u, q, '#A8A8A8', lw=lwidth)

def plot_fixed_points(plot, fixed_points, ulog=False, msize=15):
    """Adds fixed points to plots with appropriate colors based on stability.
    
    Fixed points are structured in a hiarchy of lists grouping them by environment, strategies, So 
    the list of fixed points for a community can be found as 
    fixed point list = fixed_points[environment #][strategy #].
    
    """

    for e in np.arange(fixed_points.shape[0]):
        for s in np.arange(fixed_points.shape[1]):

            hom_pts = fixed_points[e][s][0]
            het_pts = fixed_points[e][s][1]
            ext_pts = fixed_points[e][s][2]

            # Make list of all fixed points
            try:
                points = np.concatenate([np.concatenate(hom_pts), het_pts, ext_pts])
            except ValueError: # het_pts is empty causing concatinate to fail due to bad shape
                points = np.concatenate([np.concatenate(hom_pts), ext_pts])

            if ulog:
                points[:, 0] = np.log10(points[:, 0])

            for pt in points: # Note pt is already in (u, q) form

                if pt[-1] <= 0.: # unstable spirals and nodes and saddles
                    plot.plot(pt[0], pt[1], 
                            markersize=msize, color='w', marker='o', markeredgewidth=2)
                elif pt[-1] > 0.: # for stable nodes and spirals
                    if pt[0] == -np.inf:
                        plot.plot(-3, pt[1], color='k', markersize=msize, marker='o')
                    else:
                        plot.plot(pt[0], pt[1], 
                                markersize=msize, color='k', marker='o', markeredgewidth=2)
                else:
                    print 'Cannot plot point with unknown stability.', pt
                    sys.exit()

def plot_features(plot):
    """Adds features such as axes labels, colors, etc"""

    plot.set_xlim((0, 1))
    plot.set_ylim((0, 1))

#    plot.set_title('Theory', fontsize=35)
    plot.set_ylabel('Frequency of\nCooperators', fontsize=30)
    plot.set_xlabel('Total Population Density (log)', fontsize=30)
#    plot.get_xaxis().set_ticks([-3, -2, -1, 0])
    plot.tick_params(axis='both', which='major', labelsize=25)



                    
if __name__ == '__main__':
    main()

