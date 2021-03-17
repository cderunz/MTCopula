__author__ = 'BENALI Fodil'
__email__ = 'fodel.benali@gmail.com'
__copyright__ = 'Copyright (c) 2021, AdW Project'

import numpy as np
from scipy.stats import kstest

from copulas import get_instance


def select_univariate(X, candidates, margin_fit_method='AIC'):
    best_mesure = np.inf
    best_model = None
    for model in candidates:

        try:
            instance = get_instance(model)
            fitted_params = instance.fit(X)

            if (margin_fit_method == 'AIC'):
                measure = fitting_with_aic(X, instance, fitted_params)

            else:
                measure = fitting_with_ks(X, instance)

            print(model, measure)
            if measure < best_mesure:
                best_mesure = measure
                best_model = model
        except ValueError:
            # Distribution not supported
            pass

    best_instance = get_instance(best_model)

    return best_instance


def fitting_with_aic(X, instance, fitted_params):
    k = len(fitted_params)

    logLik = np.ma.masked_invalid(instance.log_probability_density(X)).sum()

    aic = 2 * k - 2 * (logLik)

    return aic


def fitting_with_ks(X, instance):
    ks, _ = kstest(X, instance.cdf)

    return ks
