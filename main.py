__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import pandas as pd

from copulas.marginals import BoundedType, MarginalDistribution
from copulas.models import TCopula
from preparation.preprocessing import data_preprocessing

univariate = MarginalDistribution(bounded=BoundedType.BOUNDED, margin_fit_method='AIC')

copula = TCopula(distribution=univariate)

bs_df = pd.read_csv('data/wdbc.csv', header=None)
features = ['id', 'diagnosis'] + ['feature ' + str(i + 1) for i in range(30)]
bs_df.columns = features
bs_df.drop('id', axis=1, inplace=True)

multivariate_variables = []

categorical_variables = ['diagnosis']

preprocesser = data_preprocessing()

data = preprocesser.prepare_data(bs_df.copy(), categorical_variables, multivariate_variables)

sample_size = data.shape[0]

copula.fit(data)

synthetic_data = copula.sample(sample_size)
