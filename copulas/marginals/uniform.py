import numpy as np
from scipy.stats import uniform

from copulas.marginals.model import BoundedType, ParametricType, ScipyModel


class UniformUnivariate(ScipyModel):
    PARAMETRIC = ParametricType.PARAMETRIC
    BOUNDED = BoundedType.BOUNDED

    MODEL_CLASS = uniform

    def _fit_constant(self, X):
        self._params = {
            'loc': np.min(X),
            'scale': np.max(X) - np.min(X)
        }

    def _fit(self, X):
        self._params = {
            'loc': np.min(X),
            'scale': np.max(X) - np.min(X)
        }

    def _is_constant(self):
        return self._params['scale'] == 0
