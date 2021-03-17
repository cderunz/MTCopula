__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import numpy as np
from scipy.stats import beta

from copulas.marginals.model import BoundedType, ParametricType, ScipyModel


class BetaUnivariate(ScipyModel):
    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED
    MODEL_CLASS = beta

    def _fit_constant(self, X):
        self._params = {
            'a': 1.0,
            'b': 1.0,
            'loc': np.unique(X)[0],
            'scale': 0.0,
        }

    def _fit(self, X):
        loc = np.min(X)
        scale = np.max(X) - loc
        a, b = beta.fit(X, floc=loc, scale=scale)[0:2]

        self._params = {
            'loc': loc,
            'scale': scale,
            'a': a,
            'b': b
        }

    def _is_constant(self):
        return self._params['scale'] == 0
