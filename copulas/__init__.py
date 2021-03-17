__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import contextlib
import importlib
from copy import deepcopy

import numpy as np
import pandas as pd

EPSILON = np.finfo(np.float32).eps


class NotFittedError(Exception):
    pass


@contextlib.contextmanager
def random_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def random_state(function):
    def wrapper(self, *args, **kwargs):
        if self.random_seed is None:
            return function(self, *args, **kwargs)

        else:
            with random_seed(self.random_seed):
                return function(self, *args, **kwargs)

    return wrapper


def get_instance(obj, **kwargs):
    instance = None
    if isinstance(obj, str):
        package, name = obj.rsplit('.', 1)
        instance = getattr(importlib.import_module(package), name)(**kwargs)

    elif isinstance(obj, type):
        instance = obj(**kwargs)

    else:
        if kwargs != dict():
            instance = obj.__class__(**kwargs)

        else:
            instance = obj.__class__(*obj.__args__, **obj.__kwargs__)

    return instance


def store_args(__init__):
    def new__init__(self, *args, **kwargs):
        args_copy = deepcopy(args)
        kwargs_copy = deepcopy(kwargs)
        __init__(self, *args, **kwargs)
        self.__args__ = args_copy
        self.__kwargs__ = kwargs_copy

    return new__init__


def get_qualified_name(_object):
    module = _object.__module__
    if hasattr(_object, '__name__'):
        _class = _object.__name__

    else:
        _class = _object.__class__.__name__

    return module + '.' + _class


def vectorize(function):
    def decorated(self, X, *args, **kwargs):
        if not isinstance(X, np.ndarray):
            return function(self, X, *args, **kwargs)

        if len(X.shape) == 1:
            X = X.reshape([-1, 1])

        if len(X.shape) == 2:
            return np.fromiter(
                (function(self, *x, *args, **kwargs) for x in X),
                np.dtype('float64')
            )

        else:
            raise ValueError('Arrays of dimensionality higher than 2 are not supported.')

    decorated.__doc__ = function.__doc__
    return decorated


def scalarize(function):
    def decorated(self, X, *args, **kwargs):
        scalar = not isinstance(X, np.ndarray)

        if scalar:
            X = np.array([X])

        result = function(self, X, *args, **kwargs)
        if scalar:
            result = result[0]

        return result

    decorated.__doc__ = function.__doc__
    return decorated


def check_valid_values(function):
    def decorated(self, X, *args, **kwargs):

        if isinstance(X, pd.DataFrame):
            W = X.values

        else:
            W = X

        if not len(W):
            raise ValueError('Your dataset is empty.')

        if W.dtype not in [np.dtype('float64'), np.dtype('int64')]:
            raise ValueError('There are non-numerical values in your data.')

        if np.isnan(W).any().any():
            raise ValueError('There are nan values in your data.')

        return function(self, X, *args, **kwargs)

    return decorated
