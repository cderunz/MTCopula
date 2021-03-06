__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

from copulas import (
    EPSILON, get_instance, get_qualified_name, random_state, store_args)
from copulas.estimator import canonical_maximum_likelihood_t
from copulas.marginals import MarginalDistribution
from copulas.models.model import Multivariate

LOGGER = logging.getLogger(__name__)
DEFAULT_DISTRIBUTION = MarginalDistribution


class TCopula(Multivariate):
    covariance = None
    columns = None
    univariates = None

    @store_args
    def __init__(self, distribution=DEFAULT_DISTRIBUTION, random_seed=None):
        self.random_seed = random_seed
        self.distribution = distribution
        self.estimator = canonical_maximum_likelihood_t()

    def __repr__(self):
        if self.distribution == DEFAULT_DISTRIBUTION:
            distribution = ''
        elif isinstance(self.distribution, type):
            distribution = 'distribution="{}"'.format(self.distribution.__name__)
        else:
            distribution = 'distribution="{}"'.format(self.distribution)

        return 'TCopula({})'.format(distribution)

    def _transform_to_normal(self, X):
        if isinstance(X, pd.Series):
            X = X.to_frame().T
        elif not isinstance(X, pd.DataFrame):
            if len(X.shape) == 1:
                X = [X]

            X = pd.DataFrame(X, columns=self.columns)

        U = list()
        for column_name, univariate in zip(self.columns, self.univariates):
            column = X[column_name]
            U.append(univariate.cdf(column.values).clip(EPSILON, 1 - EPSILON))

        return stats.t.ppf(np.column_stack(U))

    def _get_params(self, X, method):

        if method == 'pearson':
            result = self._transform_to_normal(X)
            correlation = pd.DataFrame(data=result).corr().values

        elif method == 'spearman':
            correlation = self.spearman_to_copula_correlation(X)

        else:
            correlation = self.kendall_to_copula_correlation(X)

        nu, correlation = self.estimator.fit(X, correlation)

        correlation = np.nan_to_num(correlation, nan=0.0)
        # If singular, add some noise to the diagonal
        if np.linalg.cond(correlation) > 1.0 / sys.float_info.epsilon:
            correlation = correlation + np.identity(correlation.shape[0]) * EPSILON

        return nu, correlation

    def kendall_to_copula_correlation(self, X):

        kendall_correlation = pd.DataFrame(data=X).corr(method='kendall').values
        correlation = np.sin(np.pi * 0.5 * kendall_correlation)
        return correlation

    def spearman_to_copula_correlation(self, X):

        spearman_correlation = pd.DataFrame(data=X).corr(method='spearman').values
        correlation = 2 * np.sin((np.pi / 6) * spearman_correlation)
        return correlation

    # @check_valid_values
    def fit(self, X, method='kendall'):

        LOGGER.info('Fitting %s', self)

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        columns = []
        univariates = []
        for column_name, column in X.items():
            if isinstance(self.distribution, dict):
                distribution = self.distribution.get(column_name, DEFAULT_DISTRIBUTION)

            else:
                distribution = self.distribution

            LOGGER.debug('Fitting column %s to %s', column_name, distribution)

            univariate = get_instance(distribution)

            univariate.fit(column)

            columns.append(column_name)
            univariates.append(univariate)

        self.columns = columns
        self.univariates = univariates

        LOGGER.debug('Computing covariance')
        self.degree_freedom, self.covariance = self._get_params(X, method)
        self.fitted = True

        LOGGER.debug('TCopula fitted successfully')

    def probability_density(self, X):

        self.check_fit()

        transformed = self._transform_to_normal(X)
        return self.multivariate_t_pdf(transformed, self.covariance, self.degree_freedom)

    def multivariate_t_pdf(self, X, correlation, nu):

        n, d = X.shape
        mean = np.zeros(d)
        pdfs = []
        for i in range(n):
            pdfs.append(self.estimator.pdf(X[i, :], mean, correlation, nu))

        return np.asarray(pdfs)

    def multivariate_t_logpdf(self, X, correlation, nu):

        n, d = X.shape
        mean = np.zeros(d)
        logpdfs = []
        for i in range(n):
            logpdfs.append(self.estimator.logpdf(X[i, :], mean, correlation, nu))

        return np.asarray(logpdfs)

    def cumulative_distribution(self, X):
        self.check_fit()
        transformed = self._transform_to_normal(X)
        return stats.multivariate_normal.cdf(transformed, cov=self.covariance)

    @random_state
    def sample(self, num_rows=1):

        self.check_fit()

        res = {}
        means = np.zeros(self.covariance.shape[0])
        size = (num_rows,)

        clean_cov = np.nan_to_num(self.covariance)
        Y = np.random.multivariate_normal(means, clean_cov, size=size)

        samples = np.zeros(Y.shape)
        nu = self.degree_freedom
        V = np.random.chisquare(nu, size)
        for i in range(num_rows):
            samples[i, :] = means + Y[i, :] / np.sqrt(V[i] / nu)

        for i, (column_name, univariate) in enumerate(zip(self.columns, self.univariates)):
            cdf = stats.t.cdf(samples[:, i], nu)
            res[column_name] = univariate.percent_point(cdf)

        return pd.DataFrame(data=res)

    def to_dict(self):

        self.check_fit()
        univariates = [univariate.to_dict() for univariate in self.univariates]

        return {
            'covariance': self.covariance.tolist(),
            'univariates': univariates,
            'columns': self.columns,
            'type': get_qualified_name(self),
        }

    @classmethod
    def from_dict(cls, copula_dict):

        instance = cls()
        instance.univariates = []
        instance.columns = copula_dict['columns']

        for parameters in copula_dict['univariates']:
            instance.univariates.append(MarginalDistribution.from_dict(parameters))

        instance.covariance = np.array(copula_dict['covariance'])
        instance.fitted = True

        return instance
