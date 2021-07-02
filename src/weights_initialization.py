import numpy as np

def random_initialization(n_weights, n_units, bounds=(-0.1, 0.1), **kwargs):
    lower_bnd, upper_bnd = bounds[0], bounds[1]
    if lower_bnd >= upper_bnd:
        raise ValueError(f"Lower bound must be <= than upper bound!")
    distr = np.random.uniform(low=lower_bnd, high=upper_bnd, size=(n_weights, n_units))
    if n_weights == 1:
        return distr[0]
    return distr

def glorot_initialization(n_weights, n_units, **kwargs):
    # Glorot and Bengio (2010)
    # TODO check the implementation with the paper
    distr = np.random.uniform(-np.sqrt(6 / (n_units+n_weights)), np.sqrt(6 / (n_units+n_weights)),size=(n_weights, n_units))
    if n_weights == 1:
        return distr[0]
    return distr