import numpy as np

# TODO use class constructor instead static methods if it is easier for mapping
# (string name of the function and pointer to the function)
# TODO add other activation functions if needed

class ActivationFunction:

    """All functions are defined in this way:

    Args:
        x (np.ndarray): input vector
        derivative (boolean): True -> apply the derivative, False -> apply the function

    Returns:
        np.ndarray: vector after applying activation function
    """

    @staticmethod
    def identity(x, derivative = False):
        if derivative:
            return 1.
        else:
            return x        

    @staticmethod
    def sigmoid(x, derivative = False):
        if derivative:
            _f_x = ActivationFunction.sigmoid(x)
            return _f_x * (1 - _f_x)
        else:
            return 1. / (1. + np.exp(-x))

    @staticmethod
    def tanh(x, derivative = False):
        if derivative:
            _f_x = ActivationFunction.tanh(x)
            return 1 - (_f_x * _f_x)
        else:
            return np.tanh(x)

    @staticmethod
    def relu(x, derivative = False):
        if derivative:
            return np.where(x > 0, 1, 0)
        else:
            return np.maximum(x, 0)

    @staticmethod
    def parametric_relu(x, alpha, derivative = False):
        if derivative:
            return np.where(x >= 0, 1, alpha)
        else:
            return np.maximum(x, alpha * x)