#!/usr/bin/env python2

import numpy as np
import epgg
import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot as plt

def main():

    # Set up list of environments to run over
    r = np.array([0., 1.])
    N = np.array([3, 30])
    d = np.array([0.1, 5.])
    env_ranges = np.array([r, N, d])

    # Set up list of strategies to compete
    i = np.array([1.])
    k = np.array([0.])
    n = np.array([0.])
    strat_ranges = np.array([i, k, n])

    # Initialize community type
    hd = epgg.HauertDoebeli()

    # Calculate resiliences and stabilities for randomly generated environments and strategies
    resiliences, stabilities, fix_points, env_list, strats_list = (
            epgg.random_resilience_and_stability(hd, 300, env_ranges, strat_ranges))

    stabilities = np.log10(stabilities) # this plot is best viewed with stabilities in log space

    # Calculate the trendline and correlation between resilience and evolutionary stability
    trend = trendline(resiliences, stabilities)
    corr = correlation(resiliences, stabilities)
    print 'Correlation = ', corr

    # Initialize figure and empty plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_tradeoff(ax, resiliences, stabilities, trend)

    plot_extra_features(ax)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/figure_3a.pdf')
    fig.clf()

    # Don't need to save strategies since they aren't randomized
#    np.savetxt(
#            epgg.DPATH + 'HauertDoebeli/random_environments_and_strategies_for_figure_3a.txt', 
#            env_list, header='r\tN\td', footer='correlation = {0}'.format(corr)) 




def trendline(res, stab):
    """Returns the tuple (slope, intercept) for a trendline of the data."""

    if res.size != stab.size:
        print 'Failed in trendline.'
        sys.exit()

    delta = res.size*np.sum(res*res) - np.sum(res)*np.sum(res)
    intercept = (np.sum(res*res)*np.sum(stab) - np.sum(res)*np.sum(res*stab)) / delta
    slope = (res.size*np.sum(res*stab) - np.sum(res)*np.sum(stab)) / delta
    return np.array([slope, intercept])

def correlation(res, stab):
    """Returns the r**2 values for the data."""

    if res.size != stab.size:
        print 'Failed in correlation.'
        sys.exit()

    resAve = np.sum(res) / res.size
    stabAve = np.sum(stab) / stab.size

    sigXY = np.sum((res-resAve) * (stab-stabAve))
    sigX = np.sqrt(np.sum((res-resAve) * (res-resAve)))
    sigY = np.sqrt(np.sum((stab-stabAve) * (stab-stabAve)))

    return sigXY / (sigX * sigY)

def plot_tradeoff(plot, resilience, stability, trend):
    """Adds points to plot at resilence vs stability, along with a trendline."""

    for (res, stab) in zip(resilience, stability):
        if stab == 0.:
            continue
#            plot.scatter(res, stab, s=400, marker='^', c='#1B5C62', edgecolor='k', lw='2')
        else:
            plot.scatter(res, stab, s=400, marker='^', c='#A8A8A8', edgecolor='k', lw='2')
    plot.plot([0.5*i for i in xrange(16)], [i*0.5*trend[0] + trend[1] for i in xrange(16)], 'k--', 
        lw=2)

def plot_extra_features(plot):
    """Adds features such as titles to the figure"""

    plot.set_xlim(0, 3.5)
    plot.set_ybound(-1.2, 0.2)
    
    
    plot.set_title('Theory', fontsize=35)
    plot.set_xlabel("Ecological Resilience", fontsize=30, color='#398724')
    plot.set_ylabel("Stability to\nFreeloader Invasion (log)", fontsize=30, color='#9A2936')

    plot.spines['bottom'].set_color('#398724')
    plot.spines['left'].set_color('#9A2936')

    plot.get_yaxis().set_ticks([-1.0, -0.5, 0.0])
    plot.get_xaxis().set_ticks([0, 1, 2, 3])

    plot.tick_params(axis='x', colors='#398724')
    plot.tick_params(axis='y', colors='#9A2936')
    plot.tick_params(axis='both', which='major', labelsize=25)


if __name__ == '__main__':
    main()

 
