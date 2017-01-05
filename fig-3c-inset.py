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
    d = np.array([0.8, 1.8])
    env_list = epgg.create_environments(r, N, d)

    # Set example plastic strategy 
    i = np.array([0.7])
    k = np.array([0.68])
    n = np.array([10.])
    plas_strat_list = epgg.create_strategies(i, k, n)

    # Initialize community type
    hd = epgg.HauertDoebeli()
    ghc = epgg.GlobalHillCooperators()

    # Find optimal unconditional strategy
    hd.set_environment(env_list[0])
    opt_i = np.array([hd.critical_investments()[1]])
    k = np.array([0.])
    n = np.array([0.])
    uncond_strat_list = epgg.create_strategies(opt_i, k, n)

    # Calculate resiliences
    uncond_res = epgg.calculate_ecological_resilience(hd, env_list, uncond_strat_list)
    plas_res = epgg.calculate_ecological_resilience(ghc, env_list, plas_strat_list)

    print plas_res[-1]

    # Prepare figure and empty plot
    fig = plt.figure()
    plot = fig.add_subplot(1, 1, 1)

    plot_bar_plot(plot, uncond_res, plas_res, d)

    plot_features(plot)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/fig-5b.pdf')
    fig.clf()


def plot_bar_plot(plot, uncond_res, plas_res, deathrate):
    """Returns a plot of the unconditional and plastic strategy resiliences."""

    plot.barh([0.75], np.ones(1)*deathrate[1], [0.5], color='#568D92', edgecolor='#012C30')
    plot.barh([1.75], np.ones(1)*deathrate[0], [0.5], color='#568D92',
            edgecolor='#012C30')#, tick_label=['Plastic\nstrategy', 'Unconditional\nstrategy'])

def plot_features(plot):
    """Adds features like lables, adjusts tick markes etc to plot."""

    plot.set_xlim((0., 2.))
    plot.set_ylim((0.5, 2.5))

    plot.set_xlabel('Deathrate', fontsize=30)

    plot.tick_params(axis='both', which='major', labelsize=20)
    plot.set_xticks([0.2, 0.6, 1., 1.4, 1.8])
    plot.set_yticks([1, 2]) 
    plot.set_yticklabels(['Plastic\nstrategy', 'Unconditional\nstrategy'], rotation='vertical',
            multialignment='center') 


if __name__ == '__main__':
    main()

