"""
"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import xlogy

import algopy

import matplotlib
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot

from util import eval_grad, eval_hess
from model2s import Model, get_distn, get_rates_out, objective, objective_1d


def plot_quiver():
    a = Model(1, 1)
    J = a.joint_distn
    X = []
    Y = []
    U = []
    V = []
    for ap in np.linspace(0.1, 2, 20):
        for bp in np.linspace(0.1, 2, 20):
            b = Model(ap, bp)
            alpha, beta = b.solve_EM(J)
            X.append(ap)
            Y.append(bp)
            U.append(alpha-ap)
            V.append(beta-bp)

    # init the figure
    pyplot.figure(figsize=(10, 10))

    # quiver plot
    pyplot.quiver(X, Y, U, V)
    #pyplot.show()
    pyplot.savefig('quiverplot.png')


def plot_streamplot():
    a = Model(1, 1)
    J = a.joint_distn
    gridshape = (20, 20)
    U = np.empty(gridshape, dtype=float)
    V = np.empty(gridshape, dtype=float)
    Y, X = np.mgrid[0.1:2:20j, 0.1:2:20j]
    for i in range(20):
        for j in range(20):
            ap = X[i, j]
            bp = Y[i, j]
            b = Model(ap, bp)
            alpha, beta = b.solve_EM(J)
            U[i, j] = alpha - ap
            V[i, j] = beta - bp

    # init the figure
    pyplot.figure(figsize=(10, 10))

    # stream plot
    speed = np.sqrt(U*U + V*V)
    lw = 5 * speed / speed.max()
    pyplot.streamplot(X, Y, U, V, density=0.8, color='k', linewidth=lw)
    #pyplot.show()
    pyplot.savefig('streamplot.png')


def plot_1d_em():
    a_star = 1.0
    b_star = 1.0
    t_star = 0.1
    m_star = Model(a_star * t_star, b_star * t_star)
    t_0 = 0.2
    m_0 = Model(a_star * t_0, b_star * t_0)

    print('notes:')
    print('Looking at multiple values of t.')
    print('The true value of t is', t_star)
    print('The intially guessed value of t is', t_0)
    print()

    # Get the endpoint data from the true distribution.
    J = m_star.joint_distn

    # For EM, get the expected trajectory data from the initial guess,
    # conditional on the endpoint data from the true distribution.
    distn_0, dwell_0, trans_0 = m_0.get_traj_stats(J)

    # Get summaries for the true parameter values.
    distn_x, dwell_x, trans_x = m_star.get_traj_stats(J)

    endpoint_neg_lls = []
    trajectory_neg_lls = []
    em_neg_lls = []
    em_star_neg_lls = []
    ts = np.linspace(0.01, 0.3, 60)
    for t in ts:
        a = Model(a_star * t, b_star * t)
        endpoint_neg_ll = a.get_endpoint_neg_ll(J)
        trajectory_neg_ll = a.get_trajectory_neg_ll(J)
        log_params = np.log([a.alpha, a.beta])
        em_neg_ll = objective(distn_0, dwell_0, trans_0, log_params)
        em_star_neg_ll = objective(distn_x, dwell_x, trans_x, log_params)
        endpoint_neg_lls.append(endpoint_neg_ll)
        trajectory_neg_lls.append(trajectory_neg_ll)
        em_neg_lls.append(em_neg_ll)
        em_star_neg_lls.append(em_star_neg_ll)
    print(m_0.solve_EM_1d(J))
    print(m_0.solve_EM(J))

    # convert lists to numpy arrays
    endpoint_neg_lls = np.array(endpoint_neg_lls)
    trajectory_neg_lls = np.array(trajectory_neg_lls)
    em_neg_lls = np.array(em_neg_lls)
    em_star_neg_lls = np.array(em_star_neg_lls)

    # create plots with pre-defined labels
    # http://matplotlib.org/examples/api/legend_demo.html
    fig, ax = pyplot.subplots()
    ax.plot(ts, endpoint_neg_lls, 'k--', label='endpoint neg ll')
    ax.plot(ts, trajectory_neg_lls, 'k:', label='trajectory neg ll')
    ax.plot(ts, em_neg_lls, 'k_', label='EM t0 neg ll')
    ax.plot(ts, em_star_neg_lls, 'k', label='EM star neg ll')

    legend = ax.legend(loc='upper center')

    pyplot.show()

    #pyplot.savefig('streamplot.png')


def plot_1d_em_retry():
    a_star = 1.0
    b_star = 1.0
    t_star = 0.1
    m_star = Model(a_star * t_star, b_star * t_star)
    t_0 = 0.2
    m_0 = Model(a_star * t_0, b_star * t_0)

    print('notes:')
    print('Looking at multiple values of t.')
    print('The true value of t is', t_star)
    print('The intially guessed value of t is', t_0)
    print()

    # Get the endpoint data from the true distribution.
    J = m_star.joint_distn

    # For EM, get the expected trajectory data from the initial guess,
    # conditional on the endpoint data from the true distribution.
    distn_0, dwell_0, trans_0 = m_0.get_traj_stats(J)

    # Get summaries for the true parameter values.
    #distn_x, dwell_x, trans_x = m_star.get_traj_stats(J)

    endpoint_surrogate_neg_ll = m_0.get_endpoint_neg_ll(J)
    traj_surrogate_neg_ll = m_0.get_trajectory_neg_ll(J)

    endpoint_neg_lls = []
    surrogate_neg_lls = []
    ts = np.linspace(0.01, 0.3, 60)
    for t in ts:
        # We care about the parameter value defined by t.
        a = Model(a_star * t, b_star * t)
        # Compute the usual marginal log likelihood.
        endpoint_neg_ll = a.get_endpoint_neg_ll(J)
        # Use summaries under the bogus t_0 parameter values.
        # Get the trajectory log likelihood.
        #trajectory_neg_ll = a.get_trajectory_neg_ll(J)

        log_params = np.log([a.alpha, a.beta])
        traj_neg_ll = objective(distn_0, dwell_0, trans_0, log_params)

        surrogate_neg_ll = traj_neg_ll - (
                traj_surrogate_neg_ll - endpoint_surrogate_neg_ll)

        endpoint_neg_lls.append(endpoint_neg_ll)
        surrogate_neg_lls.append(surrogate_neg_ll)
    print(m_0.solve_EM_1d(J))
    print(m_0.solve_EM(J))

    # convert lists to numpy arrays
    endpoint_neg_lls = np.array(endpoint_neg_lls)
    surrogate_neg_lls = np.array(surrogate_neg_lls)

    # create plots with pre-defined labels
    # http://matplotlib.org/examples/api/legend_demo.html
    fig, ax = pyplot.subplots()
    ax.plot(ts, endpoint_neg_lls, 'k--', label='neg ll')
    ax.plot(ts, surrogate_neg_lls, 'k:', label='surrogate')

    legend = ax.legend(loc='upper center')

    pyplot.show()



def main():
    #plot_quiver()
    #plot_streamplot()
    #plot_1d_em()
    plot_1d_em_retry()


if __name__ == '__main__':
    main()

