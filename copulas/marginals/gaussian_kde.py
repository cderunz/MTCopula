from functools import partial

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import gaussian_kde

from copulas import EPSILON, scalarize, store_args
from copulas.marginals.model import BoundedType, ParametricType, ScipyModel


class GaussianKDE(ScipyModel):
    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = gaussian_kde

    @store_args
    def __init__(self, sample_size=None, random_seed=None, bw_method=None, weights=None):
        self.random_seed = random_seed
        self._sample_size = sample_size
        self.bw_method = bw_method
        self.weights = weights

    def _get_model(self):
        dataset = self._params['dataset']
        self._sample_size = self._sample_size or len(dataset)
        return gaussian_kde(dataset, bw_method=self.bw_method, weights=self.weights)

    def _get_bounds(self):
        X = self._params['dataset']
        lower = np.min(X) - (5 * np.std(X))
        upper = np.max(X) + (5 * np.std(X))

        return lower, upper

    def probability_density(self, X):
        self.check_fit()
        return self._model.evaluate(X)

    def sample(self, n_samples=1):
        self.check_fit()
        return self._model.resample(size=n_samples)[0]

    def cumulative_distribution(self, X):
        self.check_fit()
        stdev = np.sqrt(self._model.covariance[0, 0])
        lower = ndtr((self._get_bounds()[0] - self._model.dataset) / stdev)[0]
        uppers = np.vstack([ndtr((x - self._model.dataset) / stdev)[0] for x in X])
        return (uppers - lower).dot(self._model.weights)

    def _brentq_cdf(self, value):
        # The decorator expects an instance method, but usually are decorated before being bounded
        bound_cdf = partial(scalarize(GaussianKDE.cumulative_distribution), self)

        def f(x):
            return bound_cdf(x) - value

        return f

    def percent_point(self, U):
        self.check_fit()

        if isinstance(U, np.ndarray):
            if len(U.shape) == 1:
                U = U.reshape([-1, 1])

            if len(U.shape) == 2:
                return np.fromiter(
                    (self.percent_point(u[0]) for u in U),
                    np.dtype('float64')
                )

            else:
                raise ValueError('Arrays of dimensionality higher than 2 are not supported.')

        if np.any(U > 1.0) or np.any(U < 0.0):
            raise ValueError("Expected values in range [0.0, 1.0].")

        is_one = U >= 1.0 - EPSILON
        is_zero = U <= EPSILON
        is_valid = not (is_zero or is_one)

        lower, upper = self._get_bounds()

        X = np.zeros(U.shape)
        X[is_one] = float("inf")
        X[is_zero] = float("-inf")
        X[is_valid] = brentq(self._brentq_cdf(U[is_valid]), lower, upper)

        return X

    def _fit_constant(self, X):
        sample_size = self._sample_size or len(X)
        constant = np.unique(X)[0]
        self._params = {
            'dataset': [constant] * sample_size,
        }

    def _fit(self, X):
        if self._sample_size:
            X = gaussian_kde(X, bw_method=self.bw_method,
                             weights=self.weights).resample(self._sample_size)
        self._params = {
            'dataset': X.tolist()
        }

    def _is_constant(self):
        return len(np.unique(self._params['dataset'])) == 1
