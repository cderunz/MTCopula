__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import numpy as np
import pandas as pd


class dependency_tests:

    def __init__(self):
        self.df = pd.DataFrame(
            columns=['dataset', 'approach', 'method', 'corr_rmse', 'corr_mae', 'corr_max_error', 'dimension',
                     'sample_size'])

    def corr_error(self, real_data, synthetic_data, approach='copula_gaussian', method='kendall_tau', dataset='xyz',
                   precesion=10):
        A = real_data.corr()

        B = synthetic_data.corr()

        rmse = self.root_mean_squared_error(A, B)

        mae = self.mean_absolute_error(A, B)

        max_ae = self.max_absolute_error(A, B)

        dimension = real_data.shape

        sample_size = synthetic_data.shape[0]

        self.df = self.df.append(
            {'dimension': dimension, 'sample_size': sample_size, 'corr_rmse': rmse, 'corr_mae': mae,
             'corr_max_error': max_ae, 'dataset': dataset, 'approach': approach, 'method': method}, ignore_index=True)

        return self.df

    def root_mean_squared_error(self, corr_matrix_A, corr_matrix_B, precesion=10):
        return round(np.sqrt(((corr_matrix_A - corr_matrix_B) ** 2).to_numpy().mean()), precesion)

    def mean_absolute_error(self, corr_matrix_A, corr_matrix_B, precesion=10):
        return round((abs(corr_matrix_A - corr_matrix_B)).to_numpy().mean(), precesion)

    def max_absolute_error(self, corr_matrix_A, corr_matrix_B, precesion=10):
        return round((abs(corr_matrix_A - corr_matrix_B)).to_numpy().max(), precesion)
