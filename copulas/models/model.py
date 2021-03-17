__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import pickle

import numpy as np

from copulas import NotFittedError, get_instance


class Multivariate(object):
    fitted = False

    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def fit(self, X):
        raise NotImplementedError

    def probability_density(self, X):
        raise NotImplementedError

    def log_probability_density(self, X):
        return np.log(self.probability_density(X))

    def pdf(self, X):
        return self.probability_density(X)

    def cumulative_distribution(self, X):
        raise NotImplementedError

    def cdf(self, X):
        return self.cumulative_distribution(X)

    def sample(self, num_rows=1):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError

    @classmethod
    def from_dict(cls, params):
        multivariate_class = get_instance(params['type'])
        return multivariate_class.from_dict(params)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as pickle_file:
            return pickle.load(pickle_file)

    def save(self, path):
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def check_fit(self):
        if not self.fitted:
            raise NotFittedError("This model is not fitted.")
