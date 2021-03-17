__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import numpy as np
from scipy.optimize import fmin_slsqp
from scipy.stats import truncexpon

from copulas import EPSILON, store_args
from copulas.marginals.model import BoundedType, ParametricType, ScipyModel


class TruncatedExpon(ScipyModel):
    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = truncexpon

    @store_args
    def __init__(self, min=None, max=None, random_seed=None):
        self.random_seed = random_seed
        self.min = min
        self.max = max

    def _fit_constant(self, X):
        constant = np.unique(X)[0]
        self._params = {
            'b': constant,
            'loc': constant,
            'scale': 0.0
        }

    def _fit(self, X):
        if self.min is None:
            self.min = X.min() - EPSILON

        if self.max is None:
            self.max = X.max() + EPSILON

        lower, upper, scale = self.min, self.max, X.mean()

        a = truncexpon.fit(X, (upper - lower) / scale, floc=lower, scale=scale)

        self._params = {
            'b': a[0],
            'loc': a[1],
            'scale': a[2]
        }

    def _fit2(self, X):
        if self.min is None:
            self.min = X.min() - EPSILON

        if self.max is None:
            self.max = X.max() + EPSILON

        def nnlf(params):
            loc, scale = params
            b = (self.max - loc) / scale
            return truncexpon.nnlf((b, loc, scale), X)

        initial_params = X.min(), X.mean()
        optimal = fmin_slsqp(nnlf, initial_params, iprint=False,
                             bounds=[(self.min, self.max), (0.0, (self.max - self.min) ** 2)])

        loc, scale = optimal

        b = (self.max - loc) / scale

        self._params = {
            'b': b,
            'loc': loc,
            'scale': scale
        }

    def _is_constant(self):
        return self._params['b'] == 0
