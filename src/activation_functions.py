import numpy as np

# TODO add new static method for mapping name and function

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
    def leaky_relu(x, alpha = 0.01, derivative = False):
        if derivative:
            return np.where(x >= 0, 1, alpha)
        else:
            return np.maximum(x, alpha * x)

    @staticmethod
    def init_act_function(name):
        if name == "identity":
            return ActivationFunction.identity
        elif name == "sigmoid":
            return ActivationFunction.sigmoid
        elif name == "tanh":
            return ActivationFunction.tanh
        elif name == "relu":
            return ActivationFunction.relu
        elif name == "leaky_relu":
            return ActivationFunction.leaky_relu