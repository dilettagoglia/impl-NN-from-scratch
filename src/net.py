import warnings
import json
import numpy as np
from tqdm import tqdm
from activation_functions import ActivationFunction
from error_functions import ErrorFunction
from optimizers import *

"""Weights initialization"""
# Instantiate the weights and biases of the network
def _rand_init(n_weights, n_units, **kwargs):
    lower_bnd = -0.1
    upper_bnd = 0.1
    if lower_bnd >= upper_bnd:
        raise ValueError(f"lower_lim must be <= than upper_lim")
    distr = np.random.uniform(low=lower_bnd, high=upper_bnd, size=(n_weights, n_units))
    if n_weights == 1:
        return distr[0]
    return distr

"""Layer constructor"""

class Layer:
    """
    Class that represent a layer of a neural network
    Attributes:
        inp_dim: layer's input dimension
        n_units: number of units
        act: name of the activation function for that layer
        kwargs contains other attributes for the weights initialization
    """

    def __init__(self, inp_dim, n_units, act, **kwargs):
        """ Constructor -> see parameters in the class description """
        self.weights = _rand_init(n_weights=inp_dim, n_units=n_units, **kwargs) # per ora abbiamo solo inizializzazione random
        self.biases = _rand_init(n_weights=1, n_units=n_units, **kwargs)
        self.__inp_dim = inp_dim
        self.__n_units = n_units
        self.__act = ActivationFunction.tanh
        self.__inputs = None
        self.__nets = None
        self.__outputs = None
        self.__gradient_w = None
        self.__gradient_b = None

    @property
    def inp_dim(self):
        return self.__inp_dim

    @property
    def act(self):
        return self.__act

    @property
    def n_units(self):
        return self.__n_units

    @property
    def inputs(self):
        return self.__inputs

    @property
    def nets(self):
        return self.__nets

    @property
    def outputs(self):
        return self.__outputs

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
        self.__inputs = inp # (numpy ndarray) input vector
        self.__nets = np.matmul(inp, self.weights) #  use'numpy.dot' to perform dot product of two arrays. Buth if both are 2-D arrays it is matrix multiplication and using 'matmul' or 'a@b' is preferred.
        self.__nets = np.add(self.__nets, self.biases)
        self.__outputs = self.__act.func(self.__nets)
        return self.__outputs #the vector of the current layer's outputs

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
        dOut_dNet = self.__act.deriv(self.__nets)
        delta = np.multiply(upstream_delta, dOut_dNet)
        self.__gradient_b = -delta
        self.__gradient_w = np.zeros(shape=(self.__inp_dim, self.__n_units))
        for i in range(self.__inp_dim):
            for j in range(self.__n_units):
                self.__gradient_w[i][j] = -delta[j] * self.__inputs[i]
        # the i-th row of the weights matrix corresponds to the vector formed by the i-th weight of each layer's unit
        new_upstream_delta = [np.dot(delta, self.weights[i]) for i in range(self.__inp_dim)]
        return new_upstream_delta, self.__gradient_w, self.__gradient_b


"""Neural network object"""

class Network:

    def __init__(self, input_dim, units_per_layer, act_functions, tqdm=True, **kwargs):
        """
        Args:
            input_dim: dimension of input layer
            units_per_layer: tuple of integers that indicates the number of units for each layer (input excluded)
            act_functions: list of activation function names (one for each layer)
            kwargs may contain arguments for the weights initialization
        """
        if not hasattr(units_per_layer, '__iter__'): #Return whether the object has an attribute with the given name.This is done by calling getattr(obj, name) and catching AttributeError
            units_per_layer = [units_per_layer]
            act_functions = [act_functions]
        self.__check_attributes(self, input_dim=input_dim, units_per_layer=units_per_layer, act_functions=act_functions)
        self.__optimizer = None
        self.__params = { # a dictionary with all the main parameters of the network
                            **{'input_dim': input_dim, 'units_per_layer': units_per_layer, 'act_functions': act_functions}, **kwargs}
        self.__layers = []
        layer_inp_dim = input_dim
        for i in range(len(units_per_layer)):
            # aggiungo livello alla rete (vedi classe "Layer")
            self.__layers.append(Layer(inp_dim=layer_inp_dim, n_units=units_per_layer[i], act=act_functions[i], **kwargs))
            layer_inp_dim = units_per_layer[i]

    @staticmethod #Controllo gli attributi passati al costruttore
    def __check_attributes(self, input_dim, units_per_layer, act_functions):
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("The input dimension and the number of units for all layers must be positive")
        if len(units_per_layer) != len(act_functions):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(act_functions)={len(act_functions)}")

    @property # Syntax note: the line @property is used to get the value of a private variable without using any getter methods.
    def input_dim(self):
        return self.__params['input_dim']

    @property
    def units_per_layer(self):
        return self.__params['units_per_layer']

    @property
    def layers(self):
        return self.__layers # list of net's layers (see 'Layer' objects)

    @property
    def optimizer(self):
        return self.__optimizer

    @property
    def params(self):
        return self.__params

    @property
    def weights(self):
        return [layer.weights.tolist() for layer in self.__layers]

    @weights.setter
    def weights(self, value):
        for i in range(len(value)):
            self.__layers[i].weights = value[i]

    def forward(self, inp=(2, 2, 2)):
        """
        Performs a forward pass on the whole Network (feed forward computation)

        Args:
            inp: net's input vector/matrix

        Returns:
            net's output vector/matrix
        """

        """
        First version: (before adding Layer class)
        
        a = input
        pre_activations = []
        activations = [a]
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a  = activation(z)
            pre_activations.append(z)
            activations.append(a)
        return a, pre_activations, activations
        
        """
        x = inp
        for layer in self.__layers:
            x = layer.forward_pass(x)
        return x

    def compile(self, optimizer='sgd', loss='squared', metr='bin_class_acc', lr=0.01, momentum=0., reg_type='l2', lambd=0, **kwargs): #todo: vedere loss e metrics in file Paolo
        """
        Prepares the network by assigning an optimizer to it and setting its parameters

        Args:
            optimizer: ('Optimizer' object) #todo
            loss: the type of loss function
            metr: the type of metric to track (accuracy etc)
            lr: learning rate value
            momentum:  momentum parameter
            lambd: regularization parameter
            reg_type:  regularization type
        """
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        self.__params = {**self.__params, **{'loss': loss, 'metr': metr, 'lr': lr, 'momentum': momentum,
                                             'reg_type': reg_type, 'lambd': lambd}}
        self.__optimizer = optimizers[optimizer](net=self, loss=loss, metr=metr, lr=lr, momentum=momentum, reg_type=reg_type, lambd=lambd)

    def fit(self, tr_x, tr_y, val_x, val_y, epochs=1, batch_size=1, **kwargs):
        """
        Execute the training of the network

        Args:
            tr_x: input training set
            tr_y: targets for each input training pattern
            val_x:  input validation set
            val_y:  targets for each input validation pattern
            batch_size: the size of the batch
            epochs: number of epochs
        """
        # transform sets to numpy array (if they're not already)
        tr_x, tr_y = np.array(tr_x), np.array(tr_y)
        val_x, val_y = np.array(val_x), np.array(val_y)

        n_val_examples = val_x.shape[0] # takes the dimension of validation input vector
        n_targets = val_y.shape[0]      # takes the dimension of validation target vector
        if n_val_examples != n_targets: #todo: ricontrollare questo punto
            raise AttributeError(f"Mismatching shapes in validation set {n_val_examples} {n_targets}")

        self.__params = {**self.__params, 'epochs': epochs, 'batch_size': batch_size}
        return self.__optimizer.optimize(tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size, **kwargs)

    def predict(self, inp):
        """
        Computes the outputs for a batch of patterns, useful for testing w/ a blind test set

        Args:
            inp: batch of input patterns
        Returns:
            array of net's outputs

        """

        """
        First version (before adding Layer class):
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = activation(z)
        predictions = (a > 0.5).astype(int)
        return predictions
        
        """
        inp = np.array(inp)
        inp = inp[np.newaxis, :] if len(inp.shape) < 2 else inp
        predictions = []
        for pattern in inp:
            predictions.append(self.forward(inp=pattern))
        return np.array(predictions)

    def evaluate(self, targets, metr, loss, net_outputs=None, inp=None):
        """
        Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
        or the input data ('inp'), on which the net will first compute the output.

        Args:
            targets: the targets for the input on which the net is evaluated
            metr: the metric to track for the evaluation
            loss: the loss to track for the evaluation
            net_outputs: the output of the net for a certain input
            inp: the input on which the net has to be evaluated

        Returns:
            the loss and the metric
        """
        if net_outputs is None:
            if inp is None:
                raise AttributeError("Both net_outputs and inp cannot be None")
            net_outputs = self.predict(inp)
        metr_scores = np.zeros(self.layers[-1].n_units)
        loss_scores = np.zeros(self.layers[-1].n_units)
        for x, y in zip(net_outputs, targets):
            metr_scores = np.add(metr_scores, metrics[metr].func(predicted=x, target=y)) #todo
            loss_scores = np.add(loss_scores, losses[loss].func(predicted=x, target=y)) #todo: controllare file Paolo e modificare loss e metriche
        loss_scores = np.sum(loss_scores) / len(loss_scores)
        metr_scores = np.sum(metr_scores) / len(metr_scores)
        loss_scores /= len(net_outputs)
        metr_scores /= len(net_outputs)
        return loss_scores, metr_scores

    def backprop(self, dErr_dOut, grad_net): # NB: mantenuto originale
        """
        Propagates back the error to update each layer's gradient

        Args:
            dErr_dOut: derivatives of the error wrt the outputs
            grad_net: a structure with the same topology of the neural network in question, but used to store the
                gradients. It will be updated and returned back to the caller
        Returns:
            the updated grad_net
        """
        curr_delta = dErr_dOut
        for layer_index in reversed(range(len(self.__layers))):
            curr_delta, grad_w, grad_b = self.__layers[layer_index].backward_pass(curr_delta)
            grad_net[layer_index]['weights'] = np.add(grad_net[layer_index]['weights'], grad_w)
            grad_net[layer_index]['biases'] = np.add(grad_net[layer_index]['biases'], grad_b)
        return grad_net

    def get_empty_struct(self): # NB mantenuto originale
        """ return a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        struct = np.array([{}] * len(self.__layers))
        for layer_index in range(len(self.__layers)):
            struct[layer_index] = {'weights': [], 'biases': []}
            weights_matrix = self.__layers[layer_index].weights
            weights_matrix = weights_matrix[np.newaxis, :] if len(weights_matrix.shape) < 2 else weights_matrix
            struct[layer_index]['weights'] = np.zeros(shape=weights_matrix.shape)
            struct[layer_index]['biases'] = np.zeros(shape=(len(weights_matrix[0, :])))
        return struct

    def print_topology(self): # utile???
        """ Prints the network's architecture and parameters """
        print("Model's topology:")
        print("Units per layer: ", self.__params['units_per_layer'])
        print("Activation functions: ", self.__params['act_functions'])

    def save_model(self, filename: str):
        """ Saves the model to filename """
        data = {'model_params': self.__params, 'weights': self.weights}
        with open(filename, 'w') as f:
            json.dump(data, f, indent='\t')
