__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

from math import gamma

import numpy as np
import scipy.stats as st
from numpy.linalg import inv
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma
from scipy.special import gammaln


class canonical_maximum_likelihood_t:

    def ecdf(self, X):
        n = X.size
        order = X.argsort()
        ranks = order.argsort()
        u = [(rank + 1) / (n + 1) for rank in ranks]
        return u

    def cmle(self, log_lh, theta_start=0, theta_bounds=None, optimize_method='Nelder-Mead',
             bounded_optimize_method='SLSQP', is_scalar=False):

        if is_scalar:

            if theta_bounds == None:
                return minimize_scalar(log_lh, method=optimize_method)

            return minimize_scalar(log_lh, bounds=theta_bounds, method='bounded', options={'maxiter': 200})

        if theta_bounds == None:
            return minimize(log_lh, theta_start, method=optimize_method)

        return minimize(log_lh, theta_start, method=bounded_optimize_method, bounds=[theta_bounds],
                        options={'maxiter': 50})

    def pseudo_observations(self, data):

        list_pseudo_observations = []

        for column in data.columns:
            u_i = self.ecdf(data[column])

            list_pseudo_observations.append(u_i)

        list_pseudo_observations = np.transpose(np.asarray(list_pseudo_observations))

        return list_pseudo_observations

    def place_values(self, rho):

        S = np.identity(self.dim)
        cpt = 0
        # We place rho values in the up and down triangular part of the covariance matrix
        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                S[i][j] = rho[cpt]
                S[j][i] = S[i][j]
                cpt += 1

        return S

    def invert_matrix(self, S, rho):

        # Computation of det and invert matrix
        if self.dim == 2:
            RDet = S[0, 0] * S[1, 1] - rho ** 2
            RInv = 1. / RDet * np.asarray([[S[1, 1], -rho], [-rho, S[0, 0]]])
        else:
            RDet = np.linalg.det(S)
            RInv = np.linalg.inv(S)

        return RDet, RInv

    def log_likelihood(self, params):

        nu = params[0]

        t_inv = st.t.ppf(self.pseudo_obs, df=nu)

        P = self.correlation

        # Log-likelihood
        lh = 0
        mean = np.zeros(self.pseudo_obs.shape[1])

        for i in range(self.pseudo_obs.shape[0]):
            lh += self.logpdf(t_inv[i, :], mean, P, nu)

        return -lh

    def fit(self, real_data, corrleation, **kwargs):

        self.correlation = corrleation

        self.pseudo_obs = self.pseudo_observations(real_data)

        x_start = 1.0

        res = self.cmle(self.log_likelihood, theta_start=x_start, theta_bounds=None,
                        optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
                        bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
        fitted_params = res['x']

        nu = fitted_params[0]

        return nu, self.correlation

    def multivariate_t_distribution(self, x, mu, Sigma, df):

        x = np.atleast_2d(x)  # requires x as 2d
        d = Sigma.shape[0]  # dimensionality

        numerator = gamma(1.0 * (d + df) / 2.0)

        denominator = (
            gamma(1.0 * df / 2.0) *
            np.power(df * np.pi, 1.0 * d / 2.0) *
            np.power(np.linalg.det(Sigma), 1.0 / 2.0) *
            np.power(
                1.0 + (1.0 / df) *
                np.diagonal(
                    np.dot(np.dot(x - mu, np.linalg.inv(Sigma)), (x - mu).T)
                ),
                1.0 * (d + df) / 2.0
            )
        )

        return 1.0 * numerator / denominator

    def pdf(self, x, mean, shape, df):
        return np.exp(self.logpdf(x, mean, shape, df))

    def logpdf(self, x, mean, shape, df):
        dim = mean.size

        vals, vecs = np.linalg.eigh(shape)
        logdet = np.log(vals).sum()
        valsinv = np.array([1. / v for v in vals])
        U = vecs * np.sqrt(valsinv)
        dev = x - mean
        maha = np.square(np.dot(dev, U)).sum(axis=-1)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim / 2. * np.log(df * np.pi)
        D = 0.5 * logdet
        E = -t * np.log(1 + (1. / df) * maha)

        return A - B - C - D + E
