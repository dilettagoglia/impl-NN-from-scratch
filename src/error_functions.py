import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

class ErrorFunction:
    """
    Functions for the "data term" to evaluate model error.
    """

    @staticmethod
    def squared_error(prediction, target):
        # TODO check if it is better to compute the mean inside the function 
        # be careful of types passed as parameters 
        return (np.dot(target - prediction, target - prediction))/2     # /2 usefull for the derivation
    
    @staticmethod  
    def squared_error_der(prediction, target):
        return (target - prediction)

    @staticmethod
    def euclidean_error(prediction, target):
        return np.linalg.norm(prediction - target) #L2 norm

    @staticmethod
    def euclidian_error_der(prediction, target):
        return (prediction - target) / ErrorFunction.euclidean_loss(prediction, target)


