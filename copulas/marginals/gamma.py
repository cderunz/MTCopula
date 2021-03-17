import numpy as np
from scipy.stats import gamma

from copulas.marginals.model import BoundedType, ParametricType, ScipyModel


class GammaUnivariate(ScipyModel):
    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.SEMI_BOUNDED
    MODEL_CLASS = gamma

    def _fit_constant(self, X):
        self._params = {
            'a': 0.0,
            'loc': np.unique(X)[0],
            'scale': 0.0,
        }

    def _fit(self, X):
        a, loc, scale = gamma.fit(X)
        self._params = {
            'a': a,
            'loc': loc,
            'scale': scale,
        }

    def _is_constant(self):
        return self._params['scale'] == 0
