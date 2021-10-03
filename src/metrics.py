import numpy as np

# TODO maybe also add euclidian error as a metric

class Metric:

    @staticmethod
    def binary_class_accuracy(prediction, target):
        """Compute classification accuracy based on a specific threshold

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: 1-D array with ones (pattern correctly classified) and zeros (not correctly classified)
        """
        return np.where(np.abs(prediction-target) < 0.5, 1, 0)

    @staticmethod
    def euclidian_error_accuracy(prediction, target):
        """Computes the euclidian error between the predicted output vector and the target vector 

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.float64: euclidian error loss
        """
        return np.linalg.norm(prediction - target) # L2 norm for vectors



    @staticmethod
    def init_metric(name):
        if name == 'binary_class_accuracy':
            return Metric.binary_class_accuracy
        elif name == 'euclidian_error':
            return Metric.euclidian_error_accuracy
        else:
            raise NameError(name+ " is not recognized! Check for correct names and possible metrics in init_metric!")


