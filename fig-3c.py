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
    env_ranges = np.array([r, N, d])

    # Set up list of strategies to compete
    i = np.array([0.3, 3])
    k = np.array([0., 1.])
    n = np.array([10.])
    plas_strat_ranges = np.array([i, k, n])

    # Set up list of strategies to compete
    i = np.array([0.3, 3])
    k = np.array([0.])
    n = np.array([0.])
    ucon_strat_ranges = np.array([i, k, n])

    # Initialize community type
    hd = epgg.HauertDoebeli()
    ghc = epgg.GlobalHillCooperators()

    # Calculate resiliences and stabilities for randomly generated environments and strategies
    resiliences, stabilities, fix_points, env_list, strats_list = (
            epgg.random_resilience_and_stability(ghc, 100, env_ranges, plas_strat_ranges))
    ucon_resiliences, ucon_stabilities, ucon_fix_points, ucon_env_list, ucon_strats_list = (
            epgg.random_resilience_and_stability(hd, 100, env_ranges, ucon_strat_ranges))


    stabilities = np.log10(stabilities) # this plot is best viewed with stabilities in log space
    ucon_stabilities = np.log10(ucon_stabilities) # this plot is best viewed with stabilities in log space

    # Calculate the trendline and correlation between resilience and evolutionary stability
    trend_res, trend_stab = np.array([]), np.array([])
    for (res, stab) in zip(resiliences, stabilities): # Need to remove points which beat tradeoff
        if stab != 0.: # fully stable
            trend_res, trend_stab = np.append(trend_res, res), np.append(trend_stab, stab)
    trend = trendline(trend_res, trend_stab)
    corr = correlation(trend_res, trend_stab)
    print 'Correlation = ', corr

    # Initialize figure and empty plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_tradeoff(ax, resiliences, stabilities, fix_points, trend)
#    plot_ucon_tradeoff(ax, ucon_resiliences, ucon_stabilities, ucon_fix_points)

    plot_extra_features(ax)

    fig.tight_layout()
#    fig.savefig(epgg.RPATH + 'paper/figures/fig-5a.pdf')
    plt.show()
    fig.clf()

    # Don't need to save strategies since they aren't randomized
#    np.savetxt(
#            epgg.DPATH + 'GlobalHillCooperators/random_strategies_for_figure_4b.txt', 
#            strats_list[:, 1], header='i\tk\tn', footer='correlation = {0}'.format(corr)) 



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

def plot_tradeoff(plot, resilience, stability, fix_points, trend):
    """Adds points to plot at resilence vs stability, along with a trendline."""

    for n, (res, stab) in enumerate(zip(resilience, stability)):
        if fix_points[n][1].shape[0] == 0:
            plot.scatter(res, stab, s=400, marker='^', c='#568D92', edgecolor='#012C30', lw='2')
        elif fix_points[n][1].shape[0] == 2:
            plot.scatter(res, stab, s=400, marker='^', c='#88CA76', edgecolor='#0E4100', lw='2')
        else:
            plot.scatter(res, stab, s=400, marker='^', c='#A8A8A8', edgecolor='k', lw='2')

    tline = np.ones(26)
    for n, i in enumerate(np.linspace(0, 2, 26)):
        if (i*trend[0] + trend[1]) > 0.:
            tline[n] = 0.
        else:
            tline[n] = i*trend[0] + trend[1]
    plot.plot(np.linspace(0, 2, 26), tline, 'k--', lw=6)

def plot_ucon_tradeoff(plot, resilience, stability, fix_points):
    """Adds points to plot at resilence vs stability, along with a trendline."""

    for n, (res, stab) in enumerate(zip(resilience, stability)):
        if fix_points[n][1].shape[0] == 0:
            plot.scatter(res, stab, s=400, marker='s', c='#568D92', edgecolor='#012C30', lw='2')
        elif fix_points[n][1].shape[0] == 2:
            plot.scatter(res, stab, s=400, marker='s', c='#88CA76', edgecolor='#0E4100', lw='2')
        else:
            plot.scatter(res, stab, s=400, marker='s', c='#A8A8A8', edgecolor='k', lw='2')

def plot_extra_features(plot):
    """Adds features such as titles to the figure"""

    plot.set_xlim(0.5, 2.5)
    plot.set_ybound(-1.2, 0.2)
    
    
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

 
