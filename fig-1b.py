#!/usr/bin/env python2

import sys as sys
import numpy as np
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
from phase_portrait import plot_trajectories, plot_fixed_points

def main():

    # Set up list of environments to run over
    r = np.array([7.])
    N = np.array([25.])
    d = np.array([1.5])
    env_list = epgg.create_environments(r, N, d)

    # Set up list of strategies to compete
    i = np.array([1.])
    k = np.array([0.])
    n = np.array([0.])
    strats_list = epgg.create_strategies(i, k, n)

    # Set up list of initial conditions, as these are plotted in the u, q space, it is easiest to
    # give initial conditions as such, but they must be then converted to Y, X (individual species
    # frequencies.
    u0 = np.array([0.01]) 
    q0 = np.array([0.6])
    x0_list = epgg.create_initial_frequency_list([u0, q0]) 

    u0 = np.array([0.1])
    q0 = np.array([0.02, 0.08])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([0.01, 0.03])
    q0 = np.array([1.0])
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

    plot_trajectories(plot, trajectories, ulog=True)
    plot_fixed_points(plot, fixed_points, ulog=True)

    plot_features(plot)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/figure_1b.pdf')
    fig.clf()

def plot_features(plot):
    """Adds features such as axes labels, colors, etc"""

    plot.set_xlim((-3, 0))
    plot.set_ylim((0, 1))

#    plot.set_title('Theory', fontsize=35)
    plot.set_ylabel('Frequency of\nCooperators', fontsize=30)
    plot.set_xlabel('Total Population Density', fontsize=30)
    plot.get_xaxis().set_ticks([])
    plot.get_yaxis().set_ticks([])
    plot.tick_params(axis='both', which='major', labelsize=25)



                    
if __name__ == '__main__':
    main()

