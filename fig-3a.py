#!/usr/bin/env python2

import sys as sys
import numpy as np
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt
from bifurcation import plot_xbifurcation, plot_fbifurcation
from phase_portrait import plot_trajectories, plot_fixed_points

def main():

    # Set up list of environments to run over
    r = np.array([5.])
    N = np.array([8.])
    d = np.array([0.8])
    env_list = epgg.create_environments(r, N, d)

    # Set up list of strategies to compete
    i = np.linspace(0., 1.5, 250)
    k = np.array([0.])
    n = np.array([0.])
    strats_list = epgg.create_strategies(i, k, n)

    # Set example strategy for the inset
    i0 = np.array([1.])
    k0 = np.array([0.55])
    n0 = np.array([10.])
    exm_strat = epgg.create_strategies(i0, k0, n0)

    # These are for the inset
    # Set up list of initial conditions, as these are plotted in the u, q space, it is easiest to
    # give initial conditions as such, but they must be then converted to Y, X (individual species
    # frequencies.
    u0 = np.array([0.66]) 
    q0 = np.array([0.4])
    x0_list = epgg.create_initial_frequency_list([u0, q0]) 

    u0 = np.array([0.4, 0.8])
    q0 = np.array([0.85])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([0.8])
    q0 = np.array([0.6, 0.15])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    u0 = np.array([0.35])
    q0 = np.array([0.06])
    x0_list = np.concatenate((x0_list, epgg.create_initial_frequency_list([u0, q0])))

    x0_list = np.array([[u*(1-q), u*q] for [u, q] in x0_list]) #transformation mentioned just above

    # Initialize community type
    hd = epgg.HauertDoebeli()
    ghc = epgg.GlobalHillCooperators()

    # Calculate fixed points
    fixed_points = epgg.calculate_fixed_points(hd, env_list, strats_list)
    fix_pts = epgg.calculate_fixed_points(ghc, env_list, exm_strat)
   # trajectories = epgg.evolve_trajectories(ghc, env_list, exm_strat, x0_list)
    

    # tried to extract the homogeneous points for cooperators via slicing alone, but couldn't.  The
    # fixed point array wouldn't be sliced further than getting to the array of homogeneous points
    # for cooperators and freeloaders.  I wanted to slice something like [0, :, 0, 1] for 
    # [env #, all strategies, hom pts (# 0), species (# 1 for cooperators)], but kept getting index
    # errors.
    hom_pts = np.array([pts[1] for pts in fixed_points[0][:, 0]]) 
    het_pts = fixed_points[0][:, 1]

    # Prepare figure and empty plot
#    fig = plt.figure(figsize=(8, 10))
    fig = plt.figure()
    xplot = fig.add_subplot(1, 1, 1)
#    xplot = fig.add_subplot(2, 1, 1)
#    fplot = fig.add_subplot(2, 1, 2)
#    inset = fig.add_axes([0.6, 0.7, 0.2, 0.2])

    plot_xbifurcation(xplot, i, hom_pts, het_pts)
#    plot_fbifurcation(fplot, i, hom_pts, het_pts)
    xplot_investment(xplot, ghc, env_list[0], exm_strat[0], fix_pts[0][0])
#    xplot_investment(xplot, ghc, env_list[0], exm_strat[1], fix_pts[0][1])
#    fplot_investment(fplot, ghc, env_list[0], exm_strat[0], fix_pts[0][0])
#    fplot_investment(fplot, ghc, env_list[0], exm_strat[1], fix_pts[0][1])
#    plot_trajectories(inset, trajectories, lwidth=2)
#    plot_fixed_points(inset, fix_pts, msize=5)

    # Set example strategy for the inset
    i0 = np.array([0.46])
    k0 = np.array([0.])
    n0 = np.array([0.])
    exm_strat = epgg.create_strategies(i0, k0, n0)

    fix_pts = epgg.calculate_fixed_points(hd, env_list, exm_strat)
    hdxplot_investment(xplot, hd, env_list[0], exm_strat[0], fix_pts[0][0])

    xplot_features(xplot, i0)
#    fplot_features(fplot, i0)
#    inset_features(inset)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/fig-4b.pdf')
    fig.clf()


def xplot_investment(plot, com, env, strats, fix_pts):
    """Plots the investment function on plot."""
    
    com.set_environment(env)
    com.set_strategies(strats)

    X = np.linspace(0., 1., 101)
    invst = com.investment(X)

    # Plots investment curve
    plot.plot(invst, X, lw=5, ls='-', color='#398724')

    # Plots fixed points at interesections
    hom_pt, het_pt, ext_pt = fix_pts
    try: 
        points = np.concatenate((hom_pt[1][1:], het_pt))
    except ValueError:
        points = hom_pt[1][1:]
    for pt in points: 
        if pt[-1] > 0:
            plot.plot(com.investment(np.prod(pt[:-1])), np.prod(pt[:-1]), marker='o', 
                    color='k', markersize=17)
        elif pt[-1] <= 0:
            plot.plot(com.investment(np.prod(pt[:-1])), np.prod(pt[:-1]), marker='o', 
                    color='w', markersize=17)

def hdxplot_investment(plot, com, env, strats, fix_pts):
    """Plots the investment function on plot."""
    
    com.set_environment(env)
    com.set_strategies(strats)

    X = np.linspace(0., 1., 101)
    invst = strats[1][0]*np.ones(101)

    # Plots investment curve
    plot.plot(invst, X, lw=5, ls='-', color='#a1622b')

    # Plots fixed points at interesections
    hom_pt, het_pt, ext_pt = fix_pts
    try: 
        points = np.concatenate((hom_pt[1][1:], het_pt))
    except ValueError:
        points = hom_pt[1][1:]
    for pt in points: 
        if pt[-1] > 0:
            plot.plot(strats[1][0], np.prod(pt[:-1]), marker='o', 
                    color='k', markersize=17)
        elif pt[-1] <= 0:
            plot.plot(strats[1][0], np.prod(pt[:-1]), marker='o', 
                    color='w', markersize=17)

def fplot_investment(plot, com, env, strats, fix_pts):
    """Plots the investment function on plot."""
    
    com.set_environment(env)
    com.set_strategies(strats)

    X = np.linspace(0., 1., 101)
    invst = com.investment(X)

    # Plots investment curve
#    plot.plot(invst, X, lw=5, ls='-', color='#398724')

    # Plots fixed points at interesections
    hom_pt, het_pt, ext_pt = fix_pts
    try: 
        points = np.concatenate((hom_pt[1][1:], het_pt))
    except ValueError:
        points = hom_pt[1][1:]
    for pt in points: 
        if pt[-1] > 0:
            plot.plot(strats[1][0], pt[1], marker='o', 
                    color='k', markersize=17)
        elif pt[-1] <= 0:
            plot.plot(strats[1][0], pt[1], marker='o', 
                    color='w', markersize=17)


def xplot_features(plot, invest):
    """Adds features like lables, adjusts tick markes etc to plot."""

    # Adds vertical line at example investment
#    plot.axvline(invest, lw=2, ls='-', c='k')

    plot.set_xlim((0., 1.2))
    plot.set_ylim((0., 1.))

    plot.set_ylabel('Cooperator density', fontsize=30)
    plot.set_xlabel('Investment', fontsize=30)

    plot.get_xaxis().set_ticks([0.3*i for i in xrange(5)])
    plot.get_yaxis().set_ticks([0.25*i for i in xrange(1, 5)])
    plot.tick_params(axis='both', which='major', labelsize=25)

def fplot_features(plot, invest):
    """Adds features like lables, adjusts tick markes etc to plot."""

    # Adds vertical line at example investment
#    plot.axvline(invest, lw=2, ls='-', c='k')

    plot.set_xlim((0., 1.2))
    plot.set_ylim((0., 1.))

    plot.set_ylabel('Cooperator density', fontsize=30)
    plot.set_xlabel('Investment', fontsize=30)

    plot.get_xaxis().set_ticks([0.3*i for i in xrange(5)])
    plot.get_yaxis().set_ticks([0.25*i for i in xrange(1, 5)])
    plot.tick_params(axis='both', which='major', labelsize=25)

def inset_features(plot):
    """Adds features like lables, adjusts tick markes etc to plot."""

    plot.set_ylabel('Relative frequency\nof cooperators', fontsize=10)
    plot.set_xlabel('Total Population Density', fontsize=10)

    plot.set_xlim(0, 1)
    plot.set_ylim(0, 1)

    plot.tick_params(axis='both', which='major', labelsize=0)


if __name__ == '__main__':
    main()

