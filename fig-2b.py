#!/usr/bin/env python2

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
    i = np.concatenate((np.linspace(0.15, 0.9, 10), np.linspace(0.9, 2, 7)))
    k = np.array([0., 1.])
    n = np.array([10.])
    strats_list = epgg.create_strategies(i, k, n)

    # Initialize community type
    hd = epgg.HauertDoebeli()

    # Calculate resiliences and stabilities 
    resiliences = epgg.calculate_ecological_resilience(hd, env_list, strats_list)

    stabilities = epgg.calculate_evolutionary_stability_mutation(hd, env_list, strats_list)
    stabilities = np.log10(stabilities)


    # Initialize figure and empty plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_tradeoff(ax, resiliences, stabilities)

    plot_extra_features(ax)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/fig-3c.pdf')
    fig.clf()


def plot_tradeoff(plot, resilience, stability):
    """Adds points to plot at resilence vs stability, along with a trendline."""

    for (res, stab) in zip(resilience[0], stability[0]):
        if stab == 0.:
            plot.scatter(res, stab, s=400, marker='^', c='#1B5C62', edgecolor='k', lw='2')
        else:
            plot.scatter(res, stab, s=400, marker='^', c='#A8A8A8', edgecolor='k', lw='2')

def plot_extra_features(plot):
    """Adds features such as titles to the figure"""

    plot.set_xlim(0.15, 2.25)
    plot.set_ybound(-0.1, 0.15)
    
    
    plot.set_xlabel("Ecological Resilience", fontsize=30, color='#398724')
    plot.set_ylabel("Stability to\nFreeloader Invasion (log)", fontsize=30, color='#9A2936')

    plot.spines['bottom'].set_color('#398724')
    plot.spines['left'].set_color('#9A2936')

    plot.get_yaxis().set_ticks([-1.0, -0.5, 0.0])
    plot.get_xaxis().set_ticks([0, 1, 2])

    plot.tick_params(axis='x', colors='#398724')
    plot.tick_params(axis='y', colors='#9A2936')
    plot.tick_params(axis='both', which='major', labelsize=25)


if __name__ == '__main__':
    main()

 
