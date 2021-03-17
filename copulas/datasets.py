import numpy as np
import pandas as pd
from scipy import stats

from copulas import random_seed


def sample_trivariate_xyz(size=1000, seed=42):
    with random_seed(seed):
        x = stats.beta.rvs(a=0.1, b=0.1, size=size)
        y = stats.beta.rvs(a=0.1, b=0.5, size=size)
        return pd.DataFrame({
            'x': x,
            'y': y,
            'z': np.random.normal(size=size) + y * 10
        })
