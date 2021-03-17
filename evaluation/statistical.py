__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import warnings
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.special import rel_entr
from scipy.stats import chisquare
from scipy.stats import ks_2samp
from sklearn.preprocessing import MinMaxScaler


class statistical_tests:

    def __init__(self):

        self.univariate_df = pd.DataFrame(columns=['column', 'metric', 'value', 'nature'])

    def evaluate(self, data, synthetic_data, categorical_variables=['STATION', 'VAGUE', 'JOUR', 'CIBLE']):

        stats_original_df = self.statistics(data, categorical_variables).sort_values(by=categorical_variables)

        stats_synthetic_df = self.statistics(synthetic_data, categorical_variables).sort_values(
            by=categorical_variables)

        join_df = pd.merge(stats_original_df, stats_synthetic_df, how='inner', on=categorical_variables,
                           suffixes=['_org', '_sync'])

        stats_mesures = ['min', 'max', 'mean', 'std', 'median', '95centile']

        error_df = pd.DataFrame(columns=stats_mesures)

        for i in range(len(stats_mesures)):
            error_df.iloc[:, i] = abs(join_df.iloc[:, i] - join_df.iloc[:, i + stats_original_df.columns.size])

            error_df.add_suffix('_error')

        self.error_df = self.normalize(error_df)

        return self.error_df

    def statistics(self, df, categorical_variables):

        dfa = pd.DataFrame(columns=['min', 'max', 'mean', 'std', 'median', '95centile'], index=df.index)

        columns = np.setdiff1d(df.columns, np.array(categorical_variables))

        dfa['min'] = df[columns].min(axis=1)

        dfa['max'] = df[columns].max(axis=1)

        dfa['mean'] = df[columns].mean(axis=1)

        dfa['std'] = df[columns].std(axis=1)

        dfa['median'] = df[columns].median(axis=1)

        dfa['95centile'] = df[columns].quantile(0.95, axis=1)

        # 'argmin', 'argmax',

        # dfa['argmin'] = [set(df[columns].columns[i].tolist()) for i in df[columns].values == df[columns].min(axis=1)[:,None]] #df[columns].idxmin(axis=1).astype(float)

        # dfa['argmax'] = [set(df[columns].columns[i].tolist()) for i in df[columns].values == df[columns].max(axis=1)[:,None]] #df[columns].idxmax(axis=1).astype(float)

        dfa[categorical_variables] = df[categorical_variables]

        return dfa

    def normalize(self, error_df):

        scal = MinMaxScaler()

        error_df[error_df.columns] = scal.fit_transform(error_df)

        return error_df

    def error_viz(self, df, title='Utility measures error variablity', xlabel='Statitic measures error',
                  ylabel='Variability'):

        plt.figure(figsize=(14, 6))

        plt.title(title, fontsize=18)

        plt.xlabel(xlabel, fontsize=15)

        plt.ylabel(ylabel, fontsize=15)

        plt.xticks(fontsize=12)

        return sns.boxplot(data=df, showfliers=False)

    def kolmogorov(self, real_column, synthetic_column):

        statistic, pvalue = ks_2samp(real_column, synthetic_column)

        return statistic, pvalue

    def chi_squared(self, real_column, synthetic_column):
        """This function uses the Chi-squared test to compare the distributions
        of the two categorical columns. It returns the resulting p-value so that
        a small value indicates that we can reject the null hypothesis (i.e. and
        suggests that the distributions are different).
        """
        f_obs, f_exp = self.frequencies(real_column, synthetic_column)

        if len(f_obs) == len(f_exp) == 1:

            pvalue = 1.0

        else:

            _, pvalue = chisquare(f_obs, f_exp)

        return pvalue

    """Given two iterators containing categorical data, this transforms it into
         observed/expected frequencies which can be used for statistical tests. """

    def frequencies(self, real, synthetic):
        f_obs, f_exp = [], []
        real, synthetic = Counter(real), Counter(synthetic)
        for value in synthetic:
            if value not in real:
                warnings.warn("Unexpected value %s in synthetic data." % (value,))
                real[value] += 1e-6  # Regularization to prevent NaN.
        for value in real:
            f_obs.append(synthetic[value] / sum(synthetic.values()))
            f_exp.append(real[value] / sum(real.values()))
        return f_obs, f_exp

    def get_metadata(self, data):

        metadata = {}

        for key, value in data.dtypes.items():
            metadata[str(key)] = ''.join(i for i in str(value) if not i.isdigit())

        return metadata

    def get_continous_variables(self, data):

        metadata = self.get_metadata(data)

        list_continuous = []
        list_discret = []

        for var, typos in metadata.items():

            if (typos == 'float'):

                list_continuous.append(var)
            else:
                list_discret.append(var)

        return list_continuous, list_discret

    def univariate_test(self, real_data, synthetic_data):

        categorical_type = ['object']

        metadata = self.get_metadata(real_data)

        for column_name, column_type in metadata.items():

            if (column_type in categorical_type):

                value = round(self.chi_squared(real_data[column_name], synthetic_data[column_name]), 3)
                metric = 'chi-squared'
                nature = 'p_value'
            else:
                value2, value = self.kolmogorov(real_data[column_name], synthetic_data[column_name])
                value = round(value, 3)
                metric = 'kolmogorov'
                nature = 'p_value'

            self.univariate_df = self.univariate_df.append(
                {'column': column_name, 'metric': metric, 'value': value, 'nature': nature},
                ignore_index=True)

        return self.univariate_df

    def bivariate_test(self, real_data, synthetic_data):

        list_continuous, list_discret = self.get_continous_variables(real_data)

        df = pd.DataFrame(columns=['columns', 'metric', 'value', 'nature'])

        if (len(list_continuous) > 1):

            comb = combinations(list_continuous, 2)

            for i in list(comb):
                real = real_data[[i[0], i[1]]].to_numpy()

                synthetic = synthetic_data[[i[0], i[1]]].to_numpy()

                value = self.continuous_relative_entropy(real, synthetic)

                df = df.append({'columns': i, 'metric': 'DKL_continuous', 'value': value, 'nature': 'entropy'},
                               ignore_index=True)

        if (len(list_discret) > 1):

            comb = combinations(list_discret, 2)

            for i in list(comb):
                real = real_data[[i[0], i[1]]].to_numpy()

                synthetic = synthetic_data[[i[0], i[1]]].to_numpy()

                value = self.discret_relative_entropy(real, synthetic)

                df = df.append({'columns': i, 'metric': 'DKL_discret', 'value': value, 'nature': 'entropy'},
                               ignore_index=True)

        return df

    def continuous_relative_entropy(self, real, synthetic):
        """
        This approximates the KL divergence by binning the continuous values
        to turn them into categorical values and then computing the relative
        entropy.
        """
        real[np.isnan(real)] = 0.0
        synthetic[np.isnan(synthetic)] = 0.0

        real, xedges, yedges = np.histogram2d(real[:, 0], real[:, 1])
        synthetic, _, _ = np.histogram2d(
            synthetic[:, 0], synthetic[:, 1], bins=[xedges, yedges])

        f_obs, f_exp = synthetic.flatten() + 1e-5, real.flatten() + 1e-5
        f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

        value = np.sum(rel_entr(f_obs, f_exp))

        return value

    def discret_relative_entropy(self, real, synthetic):

        assert real.shape[1] == 2, "Expected 2d data."

        assert synthetic.shape[1] == 2, "Expected 2d data."

        real = [(x[0], x[1]) for x in real]

        synthetic = [(x[0], x[1]) for x in synthetic]

        f_obs, f_exp = self.frequencies(real, synthetic)

        value = np.sum(rel_entr(f_obs, f_exp))

        return value
