#!/usr/bin/env python2

import numpy as np
import sys
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
from phase_portrait import plot_trajectories, plot_fixed_points

def main():

    # Set up environment
    r = np.array([5.])
    N = np.array([8.])
    d = np.array([0.8])
    env_list = epgg.create_environments(r, N, d)

    # Set example strategy for the inset 0
    i0 = np.array([0.6])
    k0 = np.array([0.4])
    n0 = np.array([10.])
    strat_0 = epgg.create_strategies(i0, k0, n0)

    # Set example strategy for the inset 1
    i1 = np.array([1.])
    k1 = np.array([0.55])
    n1 = np.array([10.])
    strat_1 = epgg.create_strategies(i1, k1, n1)

    # Set example strategy for the inset 2
    i2 = np.array([1.2])
    k2 = np.array([0.8])
    n2 = np.array([10.])
    strat_2 = epgg.create_strategies(i2, k2, n2)

    # Set up list of initial conditions, as these are plotted in the u, q space, it is easiest to
    # give initial conditions as such, but they must be then converted to Y, X (individual species
    # frequencies.
    u0 = np.array([-0.8, -0.6])
    u0 = np.power(10., u0)
    q0 = np.array([0.1])
    x0_list = epgg.create_initial_frequency_list([u0, q0]) 

#    u0 = np.array([-1.])
#    u0 = np.power(10., u0)
#    q0 = np.array([0.3, 0.6])
#    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([-1.2])
    u0 = np.power(10., u0)
    q0 = np.array([0.4, 0.6, 0.9])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([-1.4])
    u0 = np.power(10., u0)
    q0 = np.array([0.7, 1.])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([-1.6])
    u0 = np.power(10., u0)
    q0 = np.array([1.])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([0.])
    u0 = np.power(10., u0)
    q0 = np.array([0.1, 0.9, 1.])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    x0_list = np.array([[u*(1-q), u*q] for [u, q] in x0_list]) #transformation mentioned just above

    # Initialize community type
    ghc = epgg.GlobalHillCooperators()
    ghc.set_environment(env_list[0]) # This shouldn't change throughout

    line_0, line_1, line_2  = parameter_space_boundaries(ghc)

    fpts_0 = epgg.calculate_fixed_points(ghc, env_list, strat_0)
    ext_trajectories = epgg.evolve_trajectories(ghc, env_list, strat_0, x0_list)
    fpts_1 = epgg.calculate_fixed_points(ghc, env_list, strat_1)
    hom_trajectories = epgg.evolve_trajectories(ghc, env_list, strat_1, x0_list)
    fpts_2 = epgg.calculate_fixed_points(ghc, env_list, strat_2)
    het_trajectories = epgg.evolve_trajectories(ghc, env_list, strat_2, x0_list)

    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)
    inest_0 = fig.add_axes([0.3, 0.25, 0.2, 0.2])
    inest_1 = fig.add_axes([0.65, 0.35, 0.2, 0.2])
    inest_2 = fig.add_axes([0.55, 0.7, 0.2, 0.2])

    plot_parameter_space(plot, line_0, line_1, line_2)
    plot_trajectories(inest_0, ext_trajectories, lwidth=2, ulog=True)
    plot_fixed_points(inest_0, fpts_0, msize=5, ulog=True)
    plot_trajectories(inest_1, hom_trajectories, lwidth=2, ulog=True)
    plot_fixed_points(inest_1, fpts_1, msize=5, ulog=True)
    plot_trajectories(inest_2, het_trajectories, lwidth=2, ulog=True)
    plot_fixed_points(inest_2, fpts_2, msize=5, ulog=True)


    plot_features(plot)
    inset_features(inest_0)
    inset_features(inest_1)
    inset_features(inest_2)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/figure_4c.pdf')
    fig.clf()

def parameter_space_boundaries(com):
    """"Returns 2 arrays for the amplitudes at which the phase spaces changes."""

    # Make parameters more readible
    k_list = np.linspace(1./100, 1, 100) # This is the list of relevent k
    i_list = np.linspace(1./300, 6, 300)
    n = 10.
    
    # This is a function to return a strategy given an amplitude of investment i 
    strategy = lambda i, k: np.array([[0., 0., 0.], [i, k, n]])

    line_0 = np.array([[np.nan, np.nan]])
    line_1 = np.array([[np.nan, np.nan]])
    line_2 = np.array([[np.nan, np.nan]])

    for k in k_list:

        x_to_0 = False # Set transition flags to false
        x_to_1 = False
        x_to_2 = False

        for i in i_list:

            n_fpts = number_of_fixed_points(com, strategy(i, k))

            if n_fpts == 1: # only extinction
                continue
            elif (n_fpts == 3) and not(x_to_0): # 3 homogeneous points, 0 heterogeneous
                x_to_0 = True
                line_0 = np.concatenate((line_0, np.array([[i, k]])))
            elif (n_fpts == 4) and not(x_to_1): # 3 homogeneous points, 1 heterogeneous
                x_to_1 = True
                line_1 = np.concatenate((line_1, np.array([[i, k]])))
            elif (n_fpts == 5) and not(x_to_2): # 3 homogeneous points, 2 heterogeneous
                x_to_2 = True
                line_2 = np.concatenate((line_2, np.array([[i, k]])))

    return line_0[1:], line_1[1:], line_2[1:],

def number_of_fixed_points(com, strats):
    """Returns the number of heterogeneous and homogeneous points in the system."""
    return number_of_homogeneous_points(com, strats) + number_of_heterogeneous_points(com, strats)

def number_of_heterogeneous_points(com, strats):
    """Returns the number of het. pts for a community (with an environment) and with strat."""
    com.set_strategies(strats)
    return com.heterogeneous_fixed_points().shape[0]

def number_of_homogeneous_points(com, strats):
    """Returns the number of het. pts for a community (with an environment) and with strat."""
    com.set_strategies(strats)
    return com.homogeneous_fixed_points(1).shape[0]

def plot_parameter_space(plot, line_0, line_1, line_2):
    """Plots parameter space with appropriate boundaries and shading."""

    line_2 = line_2[:-1] # Strange discontinuity

    plot.plot(np.append(np.array([10.]), line_0[:, 0]), # just bringing the line to boundary
            np.append(line_0[0, 1], line_0[:, 1]),
            'k-', lw=3)
    plot.plot(np.append(np.array([10.]), line_1[:, 0]), # just bringing the line to boundary
            np.append(line_1[0, 1], line_1[:, 1]), 
            'k-', lw=3)
    plot.plot(np.append(np.array([10.]), line_2[:, 0]), # just bringing the line to boundary
            np.append(line_2[0, 1], line_2[:, 1]), 
            'k-', lw=3)

    k_list = np.linspace(0, 1, 101) 

    for j, k in enumerate(k_list):
        if k < line_0[0, 1]:
            strt_0 = j 
        
        if k < line_1[0, 1]:
            strt_1 = j

        if k < line_2[0, 1]:
            strt_2 = j 
        elif k == line_2[-1, 1]:
            end_2_strt_1 = np.where(line_1[:, 1] == k)[0][0]

    fill_0 = np.concatenate((
        np.dstack((10.*np.ones(strt_0+2), np.linspace(0., (strt_0+1)/101., strt_0+2)))[0], 
        line_0))

    fill_1 = np.concatenate((
        np.dstack((10.*np.ones(strt_1+2), np.linspace(0., (strt_1+1)/101., strt_1+2)))[0], 
        line_1))

    fill_2 = np.concatenate((
        np.dstack((10.*np.ones(strt_2+2), np.linspace(0., (strt_2+1)/101., strt_2+2)))[0], 
        line_2, 
        line_1[end_2_strt_1+1:]))

    plot.fill_betweenx(fill_0[:, 1], fill_0[:, 0], color='#E78691', alpha=1.)
    plot.fill_betweenx(fill_2[:, 1], fill_2[:, 0], fill_0[:, 0], color='#568D92', alpha=1.)
    plot.fill_betweenx(fill_1[:, 1], fill_1[:, 0], fill_2[:, 0], color='#88CA76', alpha=1.)
    plot.fill_betweenx(fill_1[:, 1], 10.*np.ones(fill_1[:, 0].size), fill_1[:, 0], 
            color='#D4D4D4', alpha=1)


def plot_features(plot):
    """Adds features such as axes labels, colors, etc"""

    plot.set_xlim((0, 6))
    plot.set_ylim((0, 1))

#    plot.set_title('', fontsize=35)
    plot.set_ylabel('Switch threshold (k)', fontsize=30)
    plot.set_xlabel('Investment amplitude (A)', fontsize=30)
    plot.get_xaxis().set_ticks([0, 2, 4, 6])
    plot.tick_params(axis='both', which='major', labelsize=25)

def inset_features(plot):
    """Adds features like lables, adjusts tick markes etc to plot."""

    plot.set_ylabel('Relative frequency\nof cooperators', fontsize=10)
    plot.set_xlabel('Total Population Density', fontsize=10)

    plot.set_xlim(-2, 0)
    plot.set_ylim(0, 1)

    plot.tick_params(axis='both', which='major', labelsize=0)

                    
if __name__ == '__main__':
    main()

