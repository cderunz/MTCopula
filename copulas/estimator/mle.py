import numpy as np
from scipy.optimize import minimize


class maximum_likelihood:

    def fit(self, copula, real_data, **kwargs):
        self.dim = real_data.shape[1]
        rho_start = [0.0 for i in range(int(self.dim * (self.dim - 1) / 2))]
        res, estimationData = self.mle(copula, real_data, marginals=kwargs.get('marginals', None),
                                       hyper_param=kwargs.get('hyper_param', None),
                                       hyper_param_start=kwargs.get('hyper_param_start', None),
                                       hyper_param_bounds=kwargs.get('hyper_param_bounds', None),
                                       theta_start=rho_start,
                                       optimize_method=kwargs.get('optimize_method', 'Nelder-Mead'),
                                       bounded_optimize_method=kwargs.get('bounded_optimize_method', 'SLSQP'))
        rho = res['x']

        return self.place_values(rho)

    def place_values(self, rho):

        S = np.identity(self.dim)
        cpt = 0
        # We place rho values in the up and down triangular part of the covariance matrix
        for i in range(self.dim - 1):
            for j in range(i + 1, self.dim):
                S[i][j] = rho[cpt]
                S[j][i] = S[i][j]
                cpt += 1

        return S

    def mle(self, copula, X, marginals, hyper_param, hyper_param_start=None, hyper_param_bounds=None, theta_start=[0],
            theta_bounds=None, optimize_method='Nelder-Mead', bounded_optimize_method='SLSQP'):

        marginals = []
        hyper_param = []
        for v in copula.univariates:
            params = v.to_dict()
            marginals.append(v)
            hyper_param.append(params)

        hyperParams = np.asarray(hyper_param)
        hyperOptimizeParams = np.copy(
            [dic.copy() for dic in hyperParams])  # Hyper-parameters during optimization will be stored here
        hyperStart = np.asarray(hyper_param_start)
        n, d = X.shape

        # We get the initialization vector of the optimization algorithm
        thetaOffset = len(theta_start)
        start_vector = np.repeat(0, d + thetaOffset)
        start_vector[0:thetaOffset] = theta_start
        if hyper_param_start == None:
            start_vector[thetaOffset:] = [1.0 for i in range(d)]

        # The hyper-parameters that need to be fitted
        optiVector = []
        idx = 1

        # Each element of hyperParams is a dictionary
        for k in range(len(hyperParams)):
            for key in hyperParams[k]:
                optiVector.append(hyperParams[k][key])
                # If we have a start value for this specified unknown parameter
                if hyper_param_start != None and hyperParams[k][key] != None:
                    start_vector[idx] = hyperStart[k][key]
                    idx += 1

        # The global log-likelihood to maximize
        def log_likelihood(x):
            lh = 0
            idx = 1

            for k in range(len(hyperParams)):
                for key in hyperParams[k]:
                    # We need to replace None hyper-parameters with current x value of optimization algorithm
                    if hyperParams[k][key] == None:
                        hyperOptimizeParams[k][key] = x[idx]
                        idx += 1

            # The first member : the MTCopula's density
            if thetaOffset == 1:
                lh += sum([np.log(copula.probability_density(X, x[0]))])
            else:
                a = np.log(copula.probability_density(X, self.place_values(x[0:thetaOffset])))
                lh += np.isfinite(a).sum()

            # The second member : sum of PDF

            b = [np.log(marginals[j].pdf(X.iloc[:, j])).sum() for j in range(d)]
            lh += np.isfinite(b).sum()

            return lh

        # Optimization result will be stored here
        # In case whether there are bounds conditions or not, we use different methods or arguments
        optimizeResult = None
        if hyper_param_bounds == None:
            if theta_bounds == None:
                optimizeResult = minimize(lambda x: -log_likelihood(x), start_vector, method=optimize_method)
            else:
                optiBounds = np.vstack((np.array([theta_bounds]), np.tile(np.array([None, None]), [d, 1])))
                optimizeResult = minimize(lambda x: -log_likelihood(x), start_vector, method=bounded_optimize_method,
                                          bounds=optiBounds)
        else:
            if theta_bounds == None:
                optiBounds = np.vstack((np.array([None, None]), np.tile(np.array([None, None]), [d, 1])))
                optimizeResult = minimize(lambda x: -log_likelihood(x), start_vector, method=bounded_optimize_method,
                                          bounds=optiBounds)
            else:
                optiBounds = np.vstack((np.array([theta_bounds]), hyper_param_bounds))
                optimizeResult = minimize(lambda x: -log_likelihood(x), start_vector, method=bounded_optimize_method,
                                          bounds=optiBounds)

        # We replace every None values in the hyper-parameter with estimated ones
        estimatedHyperParams = hyperParams
        idx = 1
        for k in range(len(hyperParams)):
            for key in hyperParams[k]:
                if estimatedHyperParams[k][key] == None:
                    estimatedHyperParams[k][key] = optimizeResult['x'][idx]
                    idx += 1
        return optimizeResult, estimatedHyperParams
