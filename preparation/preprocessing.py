__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import pandas as pd

from preparation.transformers import CategoricalTransformer, OneHotEncodingTransformer


class data_preprocessing:

    def prepare_data(self, data2, categorical_variables=[], multivariate_variables=[]):

        data2 = self.categorical_to_continous(data2, categorical_variables)

        data2 = self.one_to_multivariate(data2, multivariate_variables)

        return data2

    def categorical_to_continous(self, data2, categorical_variables):

        categorical_transformer = CategoricalTransformer(fuzzy=True, clip=True)

        for var in categorical_variables:
            categorical_transformer.fit(data2[var])

            data2[var] = categorical_transformer.transform(data2[var])

        return data2

    def one_to_multivariate(self, data2, multivariate_variables):

        one_hot_encoding_transformer = OneHotEncodingTransformer()

        categorical_transformer = CategoricalTransformer(fuzzy=True, clip=True)

        for var in multivariate_variables:

            one_hot_encoding_transformer.fit(data2[var])

            data2[data2[var].unique().tolist()] = pd.DataFrame(one_hot_encoding_transformer.transform(data2[var]))

            for category in data2[var].unique().tolist():
                categorical_transformer.fit(data2[category])

                data2[category] = categorical_transformer.transform(data2[category])

        data2 = data2.drop(columns=multivariate_variables)

        return data2

    def null_encoding(self, data2, replace_value='mean'):

        null_transformer = NullTransformer(fill_value=replace_value, null_column='False')

        for column in data2.columns[data2.columns != 'CIBLE']:
            null_transformer.fit(data2[column])

            data2[column] = null_transformer.transform(data2[column])

        return data2
