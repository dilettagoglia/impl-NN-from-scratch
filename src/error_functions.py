import numpy as np

# TODO add new static method for mapping name and function

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
        # TODO check if it is better to compute the mean inside the function 
        # be careful of types passed as parameters 
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
        return (target - prediction)

    @staticmethod
    def euclidean_error(prediction, target):
        """Computes the euclidian error between the predicted output vector and the target vector 

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.float64: euclidian error loss
        """
        return np.linalg.norm(prediction - target) # L2 norm

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
        return np.sum(np.where(target == 0, - np.log(1. - prediction), - np.log(prediction)))

    @staticmethod
    def binary_cross_entropy_error_der(prediction, target):
        """Computes the derivative of the binary cross-entropy

        Args:
            prediction (np.ndarray): multidimensional array of shape (n,m) of predictions for the n examples
            target (np.ndarray): multidimensional array of shape (n,m) of true value for the n examples

        Returns:
            np.ndarray: derivative of the binary cross-entropy
        """
        return np.where(target == 0, 1 / (1. - prediction), - 1. / prediction)

    @staticmethod
    def init_error_function(name):
        if name == "squared_error":
            return ErrorFunction.squared_error, ErrorFunction.squared_error_der
        elif name == "euclidian_error":
            return ErrorFunction.euclidean_error, ErrorFunction.euclidian_error_der
        elif name == "binary_cross_entropy":
            return ErrorFunction.binary_cross_entropy_error, ErrorFunction.binary_cross_entropy_error_der
