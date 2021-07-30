import numpy as np

class Regularizations:

    @staticmethod
    def lasso_l1(w,lambda_):
        """Computes lasso regularization

        Args:
            w (np.ndarray): matrix of layer's weights
            lambda_ (float): regularization hyperparameter

        Returns:
            np.float64: value of regularization/penalty term
        """
        return lambda_ * np.linalg.norm(w,1)
    
    @staticmethod
    def lasso_l1_der(w,lambda_):
        """Computes lasso regularization's derivative 

        Args:
            w (np.ndarray): matrix of layer's weights
            lambda_ (float): regularization hyperparameter

        Returns:
            np.ndarray: derivative of lasso regularization
        """
        return lambda_ * np.sign(w)

    @staticmethod
    def ridge_regression_l2(w,lambda_):
        """Computes ridge regression

        Args:
            w (np.ndarray): matrix of layer's weights
            lambda_ (float): regularization hyperparameter

        Returns:
            np.float64: value of regularization/penalty term
        """
        return 1/2 * lambda_ * np.linalg.norm(w,2) * np.linalg.norm(w,2)

    @staticmethod
    def ridge_regression_l2_der(w,lambda_):
        """Computes ridge regression's derivative 

        Args:
            w (np.ndarray): matrix of layer's weights
            lambda_ (float): regularization hyperparameter

        Returns:
            np.ndarray: derivative of ridge regression
        """
        return lambda_ * w

    @staticmethod
    def init_regularization(name):
        if name == "lasso":
            return Regularizations.lasso_l1, Regularizations.lasso_l1_der
        elif name == "ridge_regression":
            return Regularizations.ridge_regression_l2, Regularizations.ridge_regression_l2_der
        else:
            raise NameError(name+ " is not recognized! Check for correct names and possible regularizations in init_regularization!")