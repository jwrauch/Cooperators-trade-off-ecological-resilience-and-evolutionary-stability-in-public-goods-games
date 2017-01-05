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
    d = np.linspace(0.5, 1.2, 20)
    env_list = epgg.create_environments(r, N, d)

    # Initialize community type
    hd = epgg.HauertDoebeli()

    # Calculate the optimal investment and population densities for each environment
    opt_i = epgg.calculate_optimal_unconditional_investment(hd, env_list)
    opt_X = np.empty(d.size)
    i0 = np.array([opt_i[0]])
    X0 = np.empty(d.size)

    for e, env in enumerate(env_list):
        hd.set_environment(env)

        hd.set_strategies(np.array([[0., 0., 0.], [opt_i[e], 0., 0.]]))
        opt_X[e] = hd.homogeneous_fixed_points()[-1][0]

        hd.set_strategies(np.array([[0., 0., 0.], [i0, 0., 0.]]))
        X0[e] = hd.homogeneous_fixed_points()[-1][0]

    # Initialize figure and empty plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    plot_coop_den_vs_parameter(ax, opt_X, X0, d)
    plot_extra_features(ax)

    fig.tight_layout()
    plt.show()
#    fig.savefig(epgg.RPATH + 'paper/figures/fig-3d.pdf')
    fig.clf()


def plot_coop_den_vs_parameter(plot, opt_X, X0, param):
    """Returns a plot of coop(erator)_den(sity) vs param(eter) for opt(imal) and sub(optimal) i"""

    # lowest extinction deathrate for i0
    ext_d = (param[np.where(X0==0)[0][0]]+param[np.where(X0==0)[0][0]-1])/2.
    plot.axvline(ext_d, -0.5, 1.05, ls='--', c='k', lw=3)

    # shade extinction and non extinction regions
    plot.fill_betweenx(np.linspace(-0.05, 1.05, 11), ext_d*np.ones(11),
            color='#D4D4D4', alpha=0.5)
    plot.fill_betweenx(np.linspace(-0.05, 1.05, 11), ext_d*np.ones(11), 1.4*np.ones(11), 
            color='#E78691', alpha=0.5)

    plot.plot(param, opt_X, 'ow', markersize=12, markeredgewidth=3, label='optimal i')
    plot.plot(param, X0, 'ok', markersize=12, markeredgewidth=3, label='initial i')
    plot.legend(loc=7)

def plot_extra_features(plot):
    """Adds features such as titles to the figure"""

    plot.set_xlim(0.45, 1.3)
    plot.set_ybound(-0.05, 0.8)
    
    
    plot.set_xlabel("Death rate", fontsize=30)
    plot.set_ylabel("Coopertor density", fontsize=30)

    plot.get_xaxis().set_ticks([0.6, 0.8, 1., 1.2])
    plot.get_yaxis().set_ticks([0.2, 0.4, 0.6, 0.8])
    plot.tick_params(axis='both', which='major', labelsize=25)



if __name__ == '__main__':
    main()

