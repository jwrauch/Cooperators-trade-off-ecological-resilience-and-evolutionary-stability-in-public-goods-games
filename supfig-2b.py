#!/usr/bin/env python2

import numpy as np
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt

def main():

    # Set up list of environments to run over
    r = np.array([7.])
    N = np.array([25.])
    d = np.linspace(0.5, 5., 11)
    env_list = epgg.create_environments(r, N, d)

    # Set up list of strategies to compete
    i = np.array([1.])
    k = np.array([0.])
    n = np.array([0.])
    strats_list = epgg.create_strategies(i, k, n)

    # Initialize community type
    hd = epgg.HauertDoebeli()

    # Initialize arrays for ecological resilience and evolutionary success
    resilience = epgg.calculate_ecological_resilience(hd, env_list, strats_list)
    stability = epgg.calculate_evolutionary_stability_mutation(hd, env_list, strats_list)

    fig = plt.figure(figsize=(8.5, 6))

    plot_dual_resilience_and_stability(fig, resilience.flat, stability.flat, d)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/figure_2b.pdf')
    fig.clf()

def plot_dual_resilience_and_stability(fig, resilience, stability, deathrate):
    """Plots the resilience vs deathrate on one side and stability vs deathrate on the other."""

    res_plt = fig.add_subplot(1, 1, 1)
    stab_plt = res_plt.twinx()

    # Plot data
    res_plt.scatter(deathrate, resilience, 
            s=400, marker='^', facecolor='#8EC580', edgecolor='#398724', lw='2')
    res_plt.plot(deathrate, resilience, ls='--', c='#398724', ms=15)
    stab_plt.scatter(deathrate, stability, 
            s=400, marker='^', facecolor='#E0919B', edgecolor='#9A2936', lw='2')
    stab_plt.plot(deathrate, stability, ls='--', c='#9A2936')

    # Labels and tick parameters
    res_plt.set_xlabel('Death rate', fontsize=30)
    res_plt.set_ylabel('Ecological Resilience', fontsize=30, color='#398724')
    stab_plt.set_ylabel('Stability to\nFreeloader Invasion', fontsize=30, color='#9A2936')
    res_plt.set_title("Theory", fontsize=35)

    res_plt.spines['left'].set_color('#398724')
    res_plt.spines['right'].set_color('#9A2936')
    res_plt.tick_params(axis='both', which='major', labelsize=25)
    res_plt.tick_params(axis='y', colors='#398724')
    res_plt.get_xaxis().set_ticks([1, 2, 3, 4])
    res_plt.get_yaxis().set_ticks([0.5, 1., 1.5, 2., 2.5])
    stab_plt.tick_params(axis='y', which='major', labelsize=25, colors='#9A2936')

    res_plt.set_xlim(0, 4.2)
    res_plt.set_ylim(0.5, 2.6)

if __name__ == '__main__':
    main()

