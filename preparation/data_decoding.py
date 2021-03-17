__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

from preparation.transformers import CategoricalTransformer


class data_decoding:

    def __init__(self):

        self.categorical_transformer = CategoricalTransformer()

    def decode(self, data, synthetic_data, categorical_variables=[], onehot_variables=[]):

        synthetic_data = self.decoding_categorical(data, synthetic_data, categorical_variables)

        synthetic_data = self.decoding_onehot(data, synthetic_data, onehot_variables)

        return synthetic_data

    def decoding_categorical(self, data, synthetic_data, categorical_variables):

        for var in categorical_variables:
            self.categorical_transformer.fit(data[var])

            synthetic_data[var] = self.categorical_transformer.reverse_transform(synthetic_data[var])

        return synthetic_data

    def decoding_onehot(self, data, synthetic_data, onehot_variables):

        for var in onehot_variables:
            synthetic_data[var] = synthetic_data[data[var].unique()].idxmax(axis="columns")

            synthetic_data = synthetic_data.drop(columns=data[var].unique())

        return synthetic_data
