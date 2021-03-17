__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import numpy as np
import scipy.stats as st
from scipy.optimize import minimize, minimize_scalar


class canonical_maximum_likelihood:

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
                        options={'maxiter': 200})

    def pseudo_observations(self, data):

        list_pseudo_observations = []

        for column in data.columns:
            u_i = self.ecdf(data[column])

            list_pseudo_observations.append(u_i)

        list_pseudo_observations = np.transpose(np.asarray(list_pseudo_observations))

        icdf = st.norm.ppf(list_pseudo_observations)

        return icdf

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

    def log_likelihood(self, rho):

        S = self.place_values(rho)

        RDet, RInv = self.invert_matrix(S, rho)

        # Log-likelihood
        lh = 0

        for i in range(self.n):
            cDens = RDet ** (-0.5) * np.exp(-0.5 * np.dot(self.ICDF[i, :], np.dot(RInv, self.ICDF[i, :])))

            lh += np.log(cDens)

        return -lh

    def fit(self, real_data, **kwargs):

        self.dim = real_data.shape[1]

        self.ICDF = self.pseudo_observations(real_data)

        self.n = real_data.shape[0]

        rho_start = [0.0 for i in range(int(self.dim * (self.dim - 1) / 2))]

        res = self.cmle(self.log_likelihood, theta_start=rho_start, theta_bounds=None,
                        optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
                        bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))

        correlation_matrix = self.place_values(res['x'])

        return correlation_matrix
