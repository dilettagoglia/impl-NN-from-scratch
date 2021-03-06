import numpy as np
from utility import eta

class ErrorFunction:
    """
    Functions for the "data term" to evaluate model error.
    """

    @staticmethod
    def squared_error(prediction, target):
        """Computes the squared error between the predicted output vector and the target vector 

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.float64: squared error loss
        """ 
        return (np.dot(target - prediction, target - prediction))/2     # /2 usefull for the derivation
    
    @staticmethod  
    def squared_error_der(prediction, target):
        """Computes the derivative of the squared error

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: derivative of the squared error
        """
        return (prediction - target)

    @staticmethod
    def euclidean_error(prediction, target):
        """Computes the euclidian error between the predicted output vector and the target vector 

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.float64: euclidian error loss
        """
        return np.linalg.norm(prediction - target) # L2 norm for vectors

    @staticmethod
    def euclidian_error_der(prediction, target):
        """Computes the derivative of the euclidian error

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: derivative of the euclidian error
        """
        return (prediction - target) / ErrorFunction.euclidean_loss(prediction, target)

    @staticmethod
    def binary_cross_entropy_error(prediction, target):
        """Computes the binary cross-entropy/log loss between the predicted output vector and the target vector 

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.float64: binary cross-entropy loss
        """
        return np.sum(np.where(target == 0, - np.log(eta(1. - prediction)), - np.log(eta(prediction))))

    @staticmethod
    def binary_cross_entropy_error_der(prediction, target):
        """Computes the derivative of the binary cross-entropy

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: derivative of the binary cross-entropy
        """
        return np.where(target == 0, 1 / eta(1. - prediction), - 1. / eta(prediction))

    @staticmethod
    def init_error_function(name):
        if name == "squared_error":
            return ErrorFunction.squared_error, ErrorFunction.squared_error_der, name
        elif name == "euclidian_error":
            return ErrorFunction.euclidean_error, ErrorFunction.euclidian_error_der, name
        elif name == "binary_cross_entropy":
            return ErrorFunction.binary_cross_entropy_error, ErrorFunction.binary_cross_entropy_error_der, name
        else:
            raise NameError(name+ " is not recognized! Check for correct names and possible error functions in init_error_function!")
