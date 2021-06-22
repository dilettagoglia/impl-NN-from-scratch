import numpy as np

# TODO use class constructor instead static methods if easier for mapping
# (string name of the function and pointer to the function)

class Regularizations:

    @staticmethod
    def lasso_l1(w,lambda_):
        return lambda_ * np.linalg.norm(w,1)
    
    @staticmethod
    def lasso_l1_der(w,lambda_):
        return lambda_ * np.sign(w)

    @staticmethod
    def ridge_regression_l2(w,lambda_):
        return 1/2 * lambda_ * np.linalg.norm(w,2) * np.linalg.norm(w,2)

    @staticmethod
    def ridge_regression_l2_der(w,lambda_):
        return lambda_ * w
