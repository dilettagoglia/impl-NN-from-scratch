import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

class LossFunction:

    @staticmethod
    def squared_loss(prediction, target):
        # TODO check if it is better to compute the mean inside the function 
        # be careful of types passed as parameters 
        return (np.dot(target - prediction, target - prediction))/2     # /2 usefull for the derivation
    
    @staticmethod  
    def square_loss_der(prediction, target):
        return (target - prediction)

    @staticmethod
    def euclidean_loss(prediction, target):
        return np.linalg.norm(prediction - target)

    @staticmethod
    def euclidian_loss_der(prediction, target):
