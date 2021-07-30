import warnings
import json
import numpy as np
# from tqdm import tqdm
from activation_functions import ActivationFunction
from error_functions import ErrorFunction
from weights_initialization import WeightsInitialization
from layer import Layer
#from optimizers import *

class Network:

    def __init__(self, input_dim, units_per_layer, act_functions, tqdm=True, **kwargs):
        # TODO rewrite all documentation with DocString
        """
        Args:
            input_dim: dimension of input layer
            units_per_layer: tuple of integers that indicates the number of units for each layer (input excluded)
            act_functions: list of activation function names (one for each layer)
            kwargs may contain arguments for the weights initialization
        """
        # TODO decide whether to keep type checking or not
        """"if not hasattr(units_per_layer, '__iter__'): #Return whether the object has an attribute with the given name.This is done by calling getattr(obj, name) and catching AttributeError
            units_per_layer = [units_per_layer]
            act_functions = [act_functions]
        self.__check_attributes(self, input_dim=input_dim, units_per_layer=units_per_layer, act_functions=act_functions)"""
        self._params = { # a dictionary with all the main parameters of the network
                            **{'input_dim': input_dim, 'units_per_layer': units_per_layer, 'act_functions': act_functions}, **kwargs}
        self._optimizer_params = {} # a dictionary with all hyperparameters useful for SGD
        self._layers = []
        layer_inp_dim = input_dim
        for i in range(len(units_per_layer)):
            # add layer to the network (vedi classe "Layer")
            self._layers.append(Layer(inp_dim=layer_inp_dim, n_units=units_per_layer[i], act=act_functions[i], **kwargs))
            # keep the current number of neurons of this layer as number of inputs for the next layer
            layer_inp_dim = units_per_layer[i]

    # TODO decide whether to keep check attribute or not
    """@staticmethod #Controllo gli attributi passati al costruttore
    def __check_attributes(self, input_dim, units_per_layer, act_functions):
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("The input dimension and the number of units for all layers must be positive")
        if len(units_per_layer) != len(act_functions):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(act_functions)={len(act_functions)}")"""

    # Syntax note: the line @property is used to get the value of a private variable without using any getter methods.
    @property
    def input_dim(self):
        return self._params['input_dim']

    @property
    def units_per_layer(self):
        return self._params['units_per_layer']

    @property
    def layers(self):
        # list of net's layers (see 'Layer' objects)
        for i,layer in enumerate(self._layers):
            print("---------- LAYER {} ----------".format(i+1))
            layer.print_details()

    @property
    def optimizer_params(self):
        return self._optimizer_params

    @property
    def params(self):
        return self._params

    @property
    def weights(self):
        return [layer.weights.tolist() for layer in self._layers]

    @weights.setter
    def weights(self, value):
        for i in range(len(value)):
            self._layers[i].weights = value[i]

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
        for layer in self._layers:
            x = layer.forward_pass(x)
        return x

    def compile(self, optimizer='sgd', loss='squared_error', metr='binary_accuracy', lr=0.01, momentum=0., reg_type='l2', lambd=0, **kwargs):
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
        self._params = {**self._params, **{'loss': loss, 'metr': metr, 'lr': lr, 'momentum': momentum,
                                             'reg_type': reg_type, 'lambd': lambd}}
        self.__optimizer = Optimizers.optimizer(net=self, loss=loss, metr=metr, lr=lr, momentum=momentum, reg_type=reg_type, lambd=lambd)

    # TODO merge fit and optimize in optimizer class (or create another class for training)
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

        self._params = {**self._params, 'epochs': epochs, 'batch_size': batch_size}
        return self.__optimizer.optimize( #todo: !!!
            tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size, **kwargs)

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
            metr_scores = np.add(metr_scores, Metric.metr(predicted=x, target=y)) #todo controllare quali parametri in costruttore file Paolo
            loss_scores = np.add(loss_scores, ErrorFunction.loss(predicted=x, target=y))
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
        curr_delta = dErr_dOut # recall backward_pass function in Layer class, where: delta = delta_Err / delta_out
        for layer_index in reversed(range(len(self._layers))):
            curr_delta, grad_w, grad_b = self._layers[layer_index].backward_pass(curr_delta)
            grad_net[layer_index]['weights'] = np.add(grad_net[layer_index]['weights'], grad_w)
            grad_net[layer_index]['biases'] = np.add(grad_net[layer_index]['biases'], grad_b)
        return grad_net

    def get_empty_struct(self): # NB mantenuto originale
        """ return a zeroed structure with the same topology of the NN to contain all the layers' gradients """
        struct = np.array([{}] * len(self._layers)) # in each {} there is a layer
        for layer_index in range(len(self._layers)):
            struct[layer_index] = {'weights': [], 'biases': []}
            weights_matrix = self._layers[layer_index].weights
            weights_matrix = weights_matrix[np.newaxis, :] if len(weights_matrix.shape) < 2 else weights_matrix
            struct[layer_index]['weights'] = np.zeros(shape=weights_matrix.shape)
            struct[layer_index]['biases'] = np.zeros(shape=(len(weights_matrix[0, :])))
        return struct

    def print_topology(self): # utile???
        """ Prints the network's architecture and parameters """
        print("Model's topology:")
        print("Units per layer: ", self._params['units_per_layer'])
        print("Activation functions: ", self._params['act_functions'])

    def save_model(self, filename: str):
        """ Saves the model to filename """
        data = {'model_params': self._params, 'weights': self.weights}
        with open(filename, 'w') as f:
            json.dump(data, f, indent='\t')
