import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

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

    # TODO implement cross-entropy and binary cross-entropy
    @staticmethod
    def cross_entropy_error(prediction, target):

    @staticmethod
    def cross_entropy_error_der(prediction, target):



