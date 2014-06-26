"""
A two-state continuous time Markov model.

"""
from __future__ import division, print_function, absolute_import

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import expm, expm_frechet
from scipy.optimize import minimize
from scipy.special import xlogy

import algopy

from .util import eval_grad, eval_hess

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

    def get_endpoint_neg_ll(self, J_other):
        """
        Compute the log likelihood for only the endpoint distribution.

        """
        return -xlogy(J_other, self.joint_distn).sum()

    def get_trajectory_neg_ll(self, J_other):
        """
        Compute expected log likelihood for the trajectory.

        """
        distn, dwell, trans = self.get_traj_stats(J_other)
        log_params = np.log([self.alpha, self.beta])
        return objective(distn, dwell, trans, log_params)

    def get_traj_stats(self, J_other):
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

        return distn, dwell, trans

    def solve_EM_1d(self, J_other):
        """
        Use EM to compute the updated scaling parameter.

        Given initial parameter guesses self.alpha and self.beta
        and given the joint distribution J_other,
        compute the expectations of trajectory sufficient statistics
        and use these conditional expectations to compute max expected
        log likelihood estimate of a scaling factor of the parameters.

        """
        distn, dwell, trans = self.get_traj_stats(J_other)
        log_alpha = np.log(self.alpha)
        log_beta = np.log(self.beta)
        obj = partial(objective_1d, distn, dwell, trans, log_alpha, log_beta)
        grad = partial(eval_grad, obj)
        hess = partial(eval_hess, obj)
        x0 = np.array([0], dtype=float)
        result = minimize(obj, x0, jac=grad, hess=hess, method='trust-ncg')
        return np.exp(result.x[0])

    def solve_EM(self, J_other):
        """
        Use EM to compute updated parameter estimates.

        Given initial parameter guesses self.alpha and self.beta
        and given the joint distribution J_other,
        compute the expectations of trajectory sufficient statistics
        and use these conditional expectations to compute max expected
        log likelihood estimates of the parameters.

        """
        distn, dwell, trans = self.get_traj_stats(J_other)
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
    ll_distn = algopy.dot(distn, algopy.log(get_distn(params)))
    ll_dwell = -algopy.dot(get_rates_out(params), dwell)
    ll_trans_01 = trans[0, 1] * log_params[0]
    ll_trans_10 = trans[1, 0] * log_params[1]
    ll = ll_distn + ll_dwell + ll_trans_01 + ll_trans_10
    return -ll


def objective_1d(distn, dwell, trans, log_alpha, log_beta, boxed_log_t):
    """
    In this case the parameters include only a scaling factor.

    """
    expanded_log_params = np.array([log_alpha, log_beta]) + boxed_log_t[0]
    return objective(distn, dwell, trans, expanded_log_params)

