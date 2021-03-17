import numpy as np
import pandas as pd
import scipy.stats as stats
from faker import Faker

from preparation.transformers.base import BaseTransformer

MAPS = {}


class CategoricalTransformer(BaseTransformer):
    mapping = None
    intervals = None
    dtype = None

    def __init__(self, anonymize=False, fuzzy=False, clip=False):
        self.anonymize = anonymize
        self.fuzzy = fuzzy
        self.clip = clip

    def _get_faker(self):

        if isinstance(self.anonymize, (tuple, list)):
            category, *args = self.anonymize
        else:
            category = self.anonymize
            args = tuple()

        try:
            faker_method = getattr(Faker(), category)

            def faker():
                return faker_method(*args)

            return faker

        except AttributeError as attrerror:
            error = 'Category "{}" couldn\'t be found on faker'.format(self.anonymize)
            raise ValueError(error) from attrerror

    def _anonymize(self, data):

        faker = self._get_faker()
        uniques = data.unique()
        fake_data = [faker() for x in range(len(uniques))]

        mapping = dict(zip(uniques, fake_data))
        MAPS[id(self)] = mapping

        return data.map(mapping)

    @staticmethod
    def _get_intervals(data):

        frequencies = data.value_counts(dropna=False).reset_index()

        # Sort also by index to make sure that results are always the same
        name = data.name or 0
        sorted_freqs = frequencies.sort_values([name, 'index'], ascending=False)
        frequencies = sorted_freqs.set_index('index', drop=True)[name]

        start = 0
        end = 0
        elements = len(data)

        intervals = dict()
        for value, frequency in frequencies.items():
            prob = frequency / elements
            end = start + prob
            mean = (start + end) / 2
            std = (end - mean) / 24
            intervals[value] = (start, end, mean, std)
            start = end

        return intervals

    def fit(self, data):

        self.mapping = dict()
        self.dtype = data.dtype

        if isinstance(data, np.ndarray):
            data = pd.Series(data)

        if self.anonymize:
            data = self._anonymize(data)

        self.intervals = self._get_intervals(data)

    def _get_value(self, category):

        start, end, mean, std = self.intervals[category]

        min_value = (start - mean) / std

        max_value = (end - mean) / std

        if self.fuzzy:
            return stats.truncnorm.rvs(min_value, max_value, loc=mean, scale=std)

        return mean

    def transform(self, data):

        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if self.anonymize:
            data = data.map(MAPS[id(self)])

        return data.fillna(np.nan).apply(self._get_value).to_numpy()

    def _normalize(self, data):

        if self.clip:
            return data.clip(0, 1)

        return np.mod(data, 1)

    def reverse_transform(self, data):

        if not isinstance(data, pd.Series):
            if len(data.shape) > 1:
                data = data[:, 0]

            data = pd.Series(data)

        data = self._normalize(data)

        result = pd.Series(index=data.index, dtype=self.dtype)

        for category, values in self.intervals.items():
            start, end = values[:2]
            result[(start < data) & (data < end)] = category

        return result


class OneHotEncodingTransformer(BaseTransformer):
    dummy_na = None
    dummies = None

    def __init__(self, error_on_unknown=True):
        self.error_on_unknown = error_on_unknown

    @staticmethod
    def _prepare_data(data):

        if isinstance(data, list):
            data = np.array(data)

        if len(data.shape) > 2:
            raise ValueError('Unexpected format.')
        if len(data.shape) == 2:
            if data.shape[1] != 1:
                raise ValueError('Unexpected format.')

            data = data[:, 0]

        return data

    def fit(self, data):

        data = self._prepare_data(data)
        self.dummy_na = pd.isnull(data).any()
        self.dummies = list(pd.get_dummies(data, dummy_na=self.dummy_na).columns)

    def transform(self, data):

        data = self._prepare_data(data)
        dummies = pd.get_dummies(data, dummy_na=self.dummy_na)
        array = dummies.reindex(columns=self.dummies, fill_value=0).values.astype(int)
        for i, row in enumerate(array):
            if np.all(row == 0) and self.error_on_unknown:
                raise ValueError(f'The value {data[i]} was not seen during the fit stage.')

        return array

    def reverse_transform(self, data):

        if data.ndim == 1:
            data = data.reshape(-1, 1)

        indices = np.argmax(data, axis=1)
        return pd.Series(indices).map(self.dummies.__getitem__)
