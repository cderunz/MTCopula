import pickle
from abc import ABC
from enum import Enum

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from copulas import NotFittedError, get_instance, get_qualified_name, store_args
from copulas.marginals.selection import select_univariate


class ParametricType(Enum):
    NON_PARAMETRIC = 0
    PARAMETRIC = 1


class BoundedType(Enum):
    UNBOUNDED = 0
    SEMI_BOUNDED = 1
    BOUNDED = 2


class MarginalDistribution(object):
    PARAMETRIC = ParametricType.NON_PARAMETRIC
    BOUNDED = BoundedType.UNBOUNDED

    fitted = False
    _constant_value = None
    _instance = None

    @classmethod
    def _select_candidates(cls, parametric=None, bounded=None):

        candidates = list()
        for subclass in cls.__subclasses__():
            candidates.extend(subclass._select_candidates(parametric, bounded))
            if ABC in subclass.__bases__:
                continue
            if parametric is not None and subclass.PARAMETRIC != parametric:
                continue
            if bounded is not None and subclass.BOUNDED != bounded:
                continue

            candidates.append(subclass)

        return candidates

    @store_args
    def __init__(self, candidates=None, parametric=None, bounded=None, margin_fit_method='AIC', random_seed=None):
        self.candidates = candidates or self._select_candidates(parametric, bounded)
        self.margin_fit_method = margin_fit_method
        self.random_seed = random_seed

    @classmethod
    def __repr__(cls):
        return cls.__name__

    def check_fit(self):

        if not self.fitted:
            raise NotFittedError("This model is not fitted.")

    def _constant_sample(self, num_samples):

        return np.full(num_samples, self._constant_value)

    def _constant_cumulative_distribution(self, X):

        result = np.ones(X.shape)
        result[np.nonzero(X < self._constant_value)] = 0

        return result

    def _constant_probability_density(self, X):

        result = np.zeros(X.shape)
        result[np.nonzero(X == self._constant_value)] = 1

        return result

    def _constant_percent_point(self, X):

        return np.full(X.shape, self._constant_value)

    def _replace_constant_methods(self):

        self.cumulative_distribution = self._constant_cumulative_distribution
        self.percent_point = self._constant_percent_point
        self.probability_density = self._constant_probability_density
        self.sample = self._constant_sample

    def _set_constant_value(self, constant_value):

        self._constant_value = constant_value
        self._replace_constant_methods()

    def _check_constant_value(self, X):

        uniques = np.unique(X)
        if len(uniques) == 1:
            self._set_constant_value(uniques[0])

            return True

        return False

    def fit(self, X):

        self._instance = select_univariate(X, self.candidates, self.margin_fit_method)

        self._instance.fit(X)
        self.fitted = True

        sns.distplot(X, label='Empirical Distribution', hist=False)
        sns.distplot(self.sample(1000), label=str(self._instance), hist=False)
        plt.legend()
        plt.show()

    def probability_density(self, X):
        self.check_fit()
        return self._instance.probability_density(X)

    def log_probability_density(self, X):
        self.check_fit()
        if self._instance:
            return self._instance.log_probability_density(X)

        return np.log(self.probability_density(X))

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        self.check_fit()
        return self._instance.cumulative_distribution(X)

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def percent_point(self, U):
        self.check_fit()
        return self._instance.percent_point(U)

    def ppf(self, U):
        return self.percent_point(U)

    def sample(self, n_samples=1):
        self.check_fit()
        return self._instance.sample(n_samples)

    def _get_params(self):

        return self._instance._get_params()

    def _set_params(self, params):

        raise NotImplementedError()

    def to_dict(self):

        self.check_fit()

        params = self._get_params()
        if self.__class__ is MarginalDistribution:
            params['type'] = get_qualified_name(self._instance)
        else:
            params['type'] = get_qualified_name(self)

        return params

    @classmethod
    def from_dict(cls, params):

        params = params.copy()
        distribution = get_instance(params.pop('type'))
        distribution._set_params(params)
        distribution.fitted = True

        return distribution

    def save(self, path):

        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    @classmethod
    def load(cls, path):

        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)


class ScipyModel(MarginalDistribution, ABC):
    MODEL_CLASS = None

    _params = None
    _model = None

    def probability_density(self, X):

        self.check_fit()
        return self._model.pdf(X)

    def log_probability_density(self, X):

        self.check_fit()
        if hasattr(self._model, 'logpdf'):
            return self._model.logpdf(X)

        return np.log(self.probability_density(X))

    def cumulative_distribution(self, X):

        self.check_fit()
        return self._model.cdf(X)

    def percent_point(self, U):

        self.check_fit()
        return self._model.ppf(U)

    def sample(self, n_samples=1):

        self.check_fit()
        return self._model.rvs(n_samples)

    def _fit(self, X):

        raise NotImplementedError()

    def _get_model(self):
        return self.MODEL_CLASS(**self._params)

    def fit(self, X):

        if self._check_constant_value(X):
            self._fit_constant(X)
        else:
            self._fit(X)
            self._model = self._get_model()

        self.fitted = True

        return self._get_params()

    def _get_params(self):

        return self._params.copy()

    def _set_params(self, params):

        self._params = params.copy()
        if self._is_constant():
            self._replace_constant_methods()
        else:
            self._model = self._get_model()
