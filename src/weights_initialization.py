import numpy as np

class WeightsInitialization:

    @staticmethod
    def random_initialization(n_weights, n_units, bounds=(-0.1, 0.1)):
        lower_bnd, upper_bnd = bounds[0], bounds[1]
        distr = np.random.uniform(low=lower_bnd, high=upper_bnd, size=(n_weights, n_units))
        if n_weights == 1:
            # otherwise it returns an array with only element another array
            return distr[0]
        return distr

    @staticmethod
    def glorot_initialization(n_weights, n_units, **kwargs):
        # Glorot and Bengio (2010)
        # from PyTorch documentation
        distr = np.random.uniform(-np.sqrt(6 / (n_units+n_weights)), np.sqrt(6 / (n_units+n_weights)),size=(n_weights, n_units))
        if n_weights == 1:
            return distr[0]
        return distr

    @staticmethod
    def init_weights_initialization(name):
        if name == "random":
            return WeightsInitialization.random_initialization
        elif name == "glorot":
            return WeightsInitialization.glorot_initialization
        else:
            raise NameError(name+ " is not recognized! Check for correct names and possible initializations in init_weights_initialization!")