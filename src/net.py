import warnings
import json
import numpy as np
from tqdm import tqdm
from error_functions import ErrorFunction
from metrics import Metric
from layer import Layer
from training import Training

class Network:

    def __init__(self, input_dim, units_per_layer, act_functions, weights_init, **kwargs):

        """
        Network Constructor

        Args:
            input_dim (int): dimension of input layer
            units_per_layer (list): list of integers that indicates the number of units for each layer (input excluded)
            act_functions (list): list of activation function names (one for each layer)
            weights_init (string): weights initialization type name
            tqdm (bool, optional): usefull for showing progress bars. Defaults to True.
            kwargs may contain arguments for the weights initialization
        """

        self.__check_attributes(self, input_dim=input_dim, units_per_layer=units_per_layer, act_functions=act_functions)
        # a dictionary with all the main specific parameters of the network (using dict concatenation)
        self._params = {**{'input_dim': input_dim, 'units_per_layer': units_per_layer, 'act_functions': act_functions, 'weights_init': weights_init},  **kwargs}
        self._training_alg = None # training alghoritm used for optimization
        self._training_params = None # a dictionary with all useful hyperparameters for the optimizer
        self._layers = []
        layer_inp_dim = input_dim
        for i in range(len(units_per_layer)):
            # add layer to the network
            self._layers.append(Layer(inp_dim=layer_inp_dim, n_units=units_per_layer[i], act=act_functions[i], init_w_name=weights_init, **kwargs))
            # keep the current number of neurons of this layer as number of inputs for the next layer
            layer_inp_dim = units_per_layer[i]
        # self.print_topology()

    # TODO decide whether to keep check attribute or not
    @staticmethod
    def __check_attributes(self, input_dim, units_per_layer, act_functions):
        if input_dim < 1 or any(n_units < 1 for n_units in units_per_layer):
            raise ValueError("The input dimension and the number of units for all layers must be positive")
        if len(units_per_layer) != len(act_functions):
            raise AttributeError(
                f"Mismatching lengths --> len(units_per_layer)={len(units_per_layer)}; len(act_functions)={len(act_functions)}")

    # Syntax note: the line @property is used to get the value of a private variable without using any getter methods.
    @property
    def input_dim(self):
        return self._params['input_dim']

    @property
    def units_per_layer(self):
        return self._params['units_per_layer']

    @property
    def weights_init(self):
        return self._params['weights_init']

    @property
    def layers(self):
        # list of net's layers (see 'Layer' objects)
        return self._layers

    @property
    def params(self):
        return self._params

    @property
    def training_params(self):
        return self._training_params

    # TODO decide if keep this or implement in Layer class
    @property
    def weights(self):
        return [layer.weights.tolist() for layer in self._layers]

    @weights.setter
    def weights(self, value):
        for i in range(len(value)):
            self._layers[i].weights = value[i]

    def forward(self, inp):
        """
        Performs a forward pass on the whole Network (feed forward computation)

        Args:
            inp (np.ndarray): net's input vector

        Returns:
            np.ndarray: return the current output for this specific layer
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

    # TODO we can remove **kwargs from compile
    def compile(self, error_func='squared_error', metr='binary_class_accuracy', lr=0.01, lr_decay=None, limit_step=None, decay_rate=None, decay_steps=None, momentum=0., nesterov = False, reg_type='ridge_regression', lambda_=0, **kwargs):
        """
        Prepares the network by assigning an optimizer to it and setting its parameters

        Args:
            error_func (str, optional): the type of error function. Defaults to 'squared_error'.
            metr (str, optional): the type of metric to use. Defaults to 'binary_class_accuracy'.
            lr (float, optional): learning rate value. Defaults to 0.01.
            lr_decay (str, optional): type of decay for learning rate. Defaults to None.
            limit_step (int, optional): iteration number of weights update when we stop decaying. Defaults to None.
            decay_rate (float, optional): amount of decay at each stage (for exponential). Defaults to None.
            decay_steps (int, optional): length of each stage for decaying, composed of multiple iterations (steps). Defaults to None.
            momentum (float, optional): momentum parameter. Defaults to 0..
            reg_type (str, optional): regularization type. Defaults to 'ridge_regression'.
            lambda_ (int, optional): regularization parameter. Defaults to 0.

        Raises:
            ValueError: momentum must be between 0 and 1
        """
        if momentum > 1. or momentum < 0.:
            raise ValueError(f"momentum must be a value between 0 and 1. Got: {momentum}")
        
        self._training_params = {'error_func': error_func, 'metr': metr, 'lr': lr, 'lr_decay': lr_decay,
                                  'limit_step': limit_step, 'decay_rate': decay_rate,
                                  'decay_steps': decay_steps, 'momentum': momentum,'reg_type': reg_type, 'lambda_': lambda_}
        self._training_alg = Training(net=self, error_func=error_func, metr=metr, lr=lr, lr_decay=lr_decay, limit_step=limit_step,
                                     decay_rate=decay_rate, decay_steps=decay_steps,
                                     momentum=momentum, nesterov=nesterov, reg_type=reg_type, lambda_=lambda_)

    
    def fit(self, tr_x, tr_y, val_x = None, val_y = None, epochs=1, batch_size=1, strip_early_stopping=0, **kwargs):
        """
        Execute the training of the network

        Args:
            tr_x (np.ndarray): input training set
            tr_y (np.ndarray): targets for each input training pattern
            val_x (np.ndarray): input validation set
            val_y (np.ndarray): targets for each input validation patter
            epochs (int, optional): number of epochs. Defaults to 1.
            batch_size (int, optional): the size of the batch. Defaults to 1.
            strip_early_stopping (int, optional): limit of consecutive epochs to stop training when the validation error increased. Defaults to 0 (no early stopping).

        Raises:
            AttributeError: if the input set and targets dimensions do not match

        Returns:
            tuple of lists: return training error (for loss), training metric error, validation error and validation error metric for each epoch
        """
        # transform sets to numpy array (if they're not already)
        tr_x, tr_y = np.array(tr_x), np.array(tr_y)
        if val_x is not None and val_y is not None:
            val_x, val_y = np.array(val_x), np.array(val_y)
            n_val_examples = val_x.shape[0] # takes the dimension of validation input vector
            n_targets = val_y.shape[0]      # takes the dimension of validation target vector
            if n_val_examples != n_targets:
                raise AttributeError(f"Mismatching shapes in validation set {n_val_examples} {n_targets}")
        if batch_size == 'full':
            batch_size = len(tr_x)
        self._training_params = {**self._training_params, 'epochs': epochs, 'batch_size': batch_size}
        return self._training_alg.gradient_descent(
            tr_x=tr_x, tr_y=tr_y, val_x=val_x, val_y=val_y, epochs=epochs, batch_size=batch_size, strip_early_stopping=strip_early_stopping ,**kwargs)

    def predict(self, inp):
        """
        Computes the outputs for a batch of patterns, useful for testing w/ a blind test set

        Args:
            inp (list): batch of input patterns

        Returns:
            np.ndarray: list of predictions
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

    def evaluate(self, targets, metr, error_func, net_outputs=None, inp=None):
        """
        Performs an evaluation of the network based on the targets and either the pre-computed outputs ('net_outputs')
        or the input data ('inp'), on which the net will first compute the output.

        Args:
            targets: the targets for the input on which the net is evaluated
            metr (string): the metric to track for the evaluation
            error_func (string): the error function to track for the evaluation
            net_outputs (np.ndarray): the output of the net for a certain input
            inp (list): the input on which the net has to be evaluated

        Returns:
            tuple of np.ndarray: the error function and the metric
        """
        if net_outputs is None:
            if inp is None:
                raise AttributeError("Both net_outputs and inp cannot be None")
            net_outputs = self.predict(inp)
        metr_scores = np.zeros(self.layers[-1].n_units)
        error_func_scores = np.zeros(self.layers[-1].n_units)
        metric = Metric.init_metric(metr)
        error_function = ErrorFunction.init_error_function(error_func)
        for x, y in zip(net_outputs, targets):
            metr_scores = np.add(metr_scores, metric(prediction=x, target=y))
            error_func_scores = np.add(error_func_scores, error_function[0](prediction=x, target=y))
        error_func_scores = np.sum(error_func_scores) / len(error_func_scores) # TODO check this division with ML-cup dataset
        metr_scores = np.sum(metr_scores) / len(metr_scores)
        error_func_scores /= len(net_outputs)
        metr_scores /= len(net_outputs)
        return error_func_scores, metr_scores

    def backprop(self, dErr_dOut, grad_net): # NB: mantenuto originale
        """
        Propagates back the error to update each layer's gradient

        Args:
            dErr_dOut (np.ndarray): derivatives of the error wrt the outputs
            grad_net (np.ndarray): a structure with the same topology of the neural network in question, but used to store the
            gradients. It will be updated and returned back to the caller
        Returns:
            np.ndarray: the updated grad_net
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

    def print_topology(self):
        """ Prints the network's architecture and parameters """
        print("---------- Network's topology ----------")
        for i,layer in enumerate(self._layers):
            print("---------- LAYER {} ----------".format(i+1))
            layer.print_details()

    def save_model(self, filename: str):
        """ Saves the model to filename """
        data = {'model_params': self._params, 'training_params': self._training_params, 'weights': self.weights}
        with open(filename, 'w') as f:
            json.dump(data, f, indent='\t')
