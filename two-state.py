"""
"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
#from scipy.special import xlogy

import algopy

import matplotlib
#matplotlib.use('GTK')
#matplotlib.use('TkAgg')
#matplotlib.use('Agg')
#matplotlib.use('WX') #works
#matplotlib.use('QTAgg')
matplotlib.use('QT4Agg')
import matplotlib.pyplot as pyplot




# algopy boilerplate
def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))


# algopy boilerplate
def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


class Model(object):

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta
        self.nstates = 2

    @property
    def Q(self):
        a = self.alpha
        b = self.beta
        return np.array([
            [-a, a],
            [b, -b],
            ], dtype=float)

    @property
    def rates_out(self):
        a = self.alpha
        b = self.beta
        return np.array([a, b], dtype=float)

    @property
    def P(self):
        return expm(self.Q)

    @property
    def marginal_distn(self):
        a = self.alpha
        b = self.beta
        v = np.array([b, a], dtype=float)
        distn = v / v.sum()
        assert_allclose(distn.sum(), 1)
        return distn

    @property
    def joint_distn(self):
        a = self.alpha
        b = self.beta
        D = np.diag(self.marginal_distn)
        P = self.P
        J = np.dot(D, P)
        assert_allclose(J.sum(), 1)
        return J

    def solve_EM(self, J_other):
        """
        Use EM to compute updated parameter estimates.

        Given initial parameter guesses self.alpha and self.beta
        and given the joint distribution J_other,
        compute the expectations of trajectory sufficient statistics
        and use these conditional expectations to compute max expected
        log likelihood estimates of the parameters.

        """
        n = self.nstates

        # compute the observed initial distribution
        distn = J_other.sum(axis=1)

        # compute conditional expected dwell times
        dwell = np.zeros(n)
        for i in range(n):
            E = np.zeros((n, n), dtype=float)
            E[i, i] = 1
            interact = expm_frechet(self.Q, E, compute_expm=False)
            dwell[i] = (J_other * interact / self.P).sum()
        assert_allclose(dwell.sum(), 1)

        # compute conditional expected transition counts
        trans = np.zeros((n, n), dtype=float)
        for i in range(n):
            E = np.zeros((n, n), dtype=float)
            E[i, 1-i] = 1
            interact = expm_frechet(self.Q, self.Q*E, compute_expm=False)
            trans[i, 1-i] = (J_other * interact / self.P).sum()

        # minimize
        obj = partial(objective, distn, dwell, trans)
        grad = partial(eval_grad, obj)
        hess = partial(eval_hess, obj)
        exp_x0 = np.array([self.alpha, self.beta])
        x0 = np.log(exp_x0)
        result = minimize(obj, x0, jac=grad, hess=hess, method='trust-ncg')
        log_params = result.x
        params = np.exp(log_params)
        return params


def get_distn(params):
    n = 2
    distn = algopy.zeros_like(params)
    distn[0] = params[1] / params.sum()
    distn[1] = params[0] / params.sum()
    return distn


def get_rates_out(params):
    return params


def objective(distn, dwell, trans, log_params):
    """
    Get the expected negative log likelihood.

    This is a helper function for the EM.

    """
    params = algopy.exp(log_params)
    ll_distn = algopy.dot(distn, algopy.log(get_distn(params))).sum()
    ll_dwell = -algopy.dot(get_rates_out(params), dwell)
    ll_trans_01 = trans[0, 1] * log_params[0]
    ll_trans_10 = trans[1, 0] * log_params[1]
    ll = ll_distn + ll_dwell + ll_trans_01 + ll_trans_10
    return -ll


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


def main():
    plot_quiver()
    plot_streamplot()


if __name__ == '__main__':
    main()

