import numpy as np

class Metric:

    @staticmethod
    def binary_accuracy(prediction, target):
        """Compute classification accuracy based on a specific threshold

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: 1-D array with ones (pattern correctly classified) and zeros (not correctly classified)
        """
        # TODO check the return value when we'll develop the code for NN
        return np.where(np.abs(prediction-target) < 0.5, 1, 0)

    @staticmethod
    def init_metric(name):
        if name == 'binary_accuracy':
            return Metric.binary_accuracy


