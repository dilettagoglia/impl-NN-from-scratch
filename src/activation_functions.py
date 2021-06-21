import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

class ActivationFunction:

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def identity_der(x):
        return 1.

    @staticmethod
    def sigmoid(x):
        return 1. / (1. + np.exp(-x))

    @staticmethod
    def sigmoid_der(x):
        _f_x = ActivationFunction.sigmoid(x)
        return _f_x * (1 - _f_x)

    @staticmethod
    def tanh(x):
        return np.tanh(x)

    @staticmethod
    def tanh_der(x):
        _f_x = ActivationFunction.tanh(x)
        return 1 - (_f_x * _f_x)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0)

    @staticmethod
    def relu_der(x):
        return np.where(x > 0, 1, 0)

    @staticmethod
    def parametric_relu(x, alpha):
        return np.maximum(x, alpha * x)

    @staticmethod
    def parametric_relu_der(x, alpha):
        return np.where(x > 0, 1, alpha)