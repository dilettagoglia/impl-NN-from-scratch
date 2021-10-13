import unittest
import numpy as np
from src.activation_functions import ActivationFunction
import torch


class TestFunctions(unittest.TestCase):

    def test_activation_functions(self):
        pytorch_act_function= torch.nn.Sigmoid()
        x = np.random.uniform(low=-5., high=6., size=5) # vector of random number in range [-5,5]
        np.testing.assert_array_max_ulp(ActivationFunction.identity(x), x)
        np.testing.assert_array_max_ulp(ActivationFunction.sigmoid(x), pytorch_act_function(torch.from_numpy(x)))
        pytorch_act_function= torch.nn.Tanh()
        np.testing.assert_array_almost_equal(ActivationFunction.tanh(x), pytorch_act_function(torch.from_numpy(x)))
        pytorch_act_function= torch.nn.ReLU()
        np.testing.assert_array_almost_equal(ActivationFunction.relu(x), pytorch_act_function(torch.from_numpy(x)))
        pytorch_act_function= torch.nn.LeakyReLU()
        np.testing.assert_array_almost_equal(ActivationFunction.leaky_relu(x), pytorch_act_function(torch.from_numpy(x)))

if __name__ == '__main__':
    unittest.main()    