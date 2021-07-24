import numpy as np
from activation_functions import ActivationFunction
from weights_initialization import WeightsInitialization

"""Layer constructor"""

class Layer:
    """
    Class that represent a layer of a neural network
    Attributes:
        inp_dim: layer's input dimension
        n_units: number of units
        act: name of the activation function for that layer
        init_w_name: name of the type of weights initialization
        kwargs contains other attributes for the weights initialization
    """

    def __init__(self, inp_dim, n_units, act, init_w_name, **kwargs):
        """ Constructor -> see parameters in the class description """
        # for weights init and activation function we use a dict with key (name) and value (function)
        self.init_weights_type = {init_w_name: WeightsInitialization.init_weights_initialization(init_w_name)}
        self.weights = self.init_weights_type[init_w_name](n_weights=inp_dim, n_units=n_units, **kwargs)
        self.biases = self.init_weights_type[init_w_name](n_weights=1, n_units=n_units, **kwargs)
        self._inp_dim = inp_dim
        self._n_units = n_units
        self._act = {act: ActivationFunction.init_act_function(act)}
        self._inputs = None
        self._nets = None
        self._outputs = None
        self._gradient_w = None
        self._gradient_b = None

    @property
    def inp_dim(self):
        return self._inp_dim

    @property
    def act(self):
        # return self.__act
        for x in self._act:
            return x

    @property
    def n_units(self):
        return self._n_units

    @property
    def inputs(self):
        return self._inputs

    @property
    def nets(self):
        return self._nets

    @property
    def outputs(self):
        return self._outputs

    def forward_pass(self, inp):
        """
        Performs the forward pass on the current layer.

            Forward propagation (first version - 1 hidden layer):

            # Initialize the parameters to random values. We need to learn these.
            np.random.seed(0)
            W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
            b1 = np.zeros((1, nn_hdim))
            W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
            b2 = np.zeros((1, nn_output_dim))

            # z_i is the input of layer i and a_i is the output of layer i after applying the activation function. W_1, b_1, W_2, b_2 are parameters of our network, which we need to learn from our training data.
            z1 = X.dot(W1) + b1     # multiply weights and add bias
            a1 = np.tanh(z1)        # activation function
            z2 = a1.dot(W2) + b2    # same for
            exp_scores = np.exp(z2) # softmax

        """
        self._inputs = inp # (numpy ndarray) input vector
        self._nets = np.matmul(inp, self.weights) #  use'numpy.dot' to perform dot product of two arrays. Buth if both are 2-D arrays it is matrix multiplication and using 'matmul' or 'a@b' is preferred.
        self._nets = np.add(self._nets, self.biases)
        self._outputs = self._act.func(self._nets)
        return self._outputs #the vector of the current layer's outputs

    def backward_pass(self, upstream_delta):
        """
        Sets the layer's gradients
        Multiply (dot product) already the delta for the current layer's weights in order to have it ready for the
                    previous layer (that does not have access to this layer's weights) that will execute this method in the
                    next iteration of Network.backprop()

        Args:
            upstream_delta: for hidden layers, delta = dot_prod(delta_next, w_next) * dOut_dNet

        Returns:
            new_upstream_delta: delta already multiplied (dot product) by the current layer's weights
            gradient_w: gradient wrt weights
            gradient_b: gradient wrt biases
        """
        dOut_dNet = self._act.deriv(self.__nets)
        delta = np.multiply(upstream_delta, dOut_dNet)
        self.gradient_b = -delta
        self.gradient_w = np.zeros(shape=(self._inp_dim, self._n_units))
        for i in range(self._inp_dim):
            for j in range(self._n_units):
                self._gradient_w[i][j] = -delta[j] * self.inputs[i]
        # the i-th row of the weights matrix corresponds to the vector formed by the i-th weight of each layer's unit
        new_upstream_delta = [np.dot(delta, self.weights[i]) for i in range(self._inp_dim)]
        return new_upstream_delta, self._gradient_w, self._gradient_b
