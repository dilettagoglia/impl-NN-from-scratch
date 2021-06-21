import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

class LossFunction:

    @staticmethod
    def square_loss(prediction, target):
        #TODO see if it is better to compute the mean inside the function 
        