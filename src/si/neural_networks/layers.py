import copy
import numpy as np
from abc import ABCMeta, abstractmethod


class Layer(metaclass=ABCMeta):
    """
    Base class for neural network layers.
    Defines the common interface that all layers must implement.
    """

    @abstractmethod
    def forward_propagation(self, input, training):
        """
        Computes the output of the layer given the input.

        Parameters
        ----------
        input: np.ndarray
            The input to the layer.
        training: bool
            Whether the layer is in training mode or not.

        Returns
        -------
        output: np.ndarray
            The output of the layer.
        """

    @abstractmethod
    def backward_propagation(self, output_error):
        """
        Back propagates the output error through the layer.

        Parameters
        ----------
        output_error: np.ndarray
            The error coming from the next layer.

        Returns
        -------
        input_error: np.ndarray
            The error to propagate to the previous layer.
        """

    @abstractmethod
    def output_shape(self):
        """
        Returns the output shape of the layer.
        """

    @abstractmethod
    def parameters(self):
        """
        Returns the number of parameters of the layer.
        """

    def set_input_shape(self, input_shape):
        """
        Sets the input shape of the layer.
        """
        self._input_shape = input_shape

    def input_shape(self):
        """
        Returns the input shape of the layer.
        """
        return self._input_shape

    def layer_name(self):
        """
        Returns the name of the layer (its class name).
        """
        return self.__class__.__name__


class DenseLayer(Layer):
    """
    Fully connected (dense) layer. Each neuron connects to every
    input feature.

    Parameters
    ----------
    n_units: int
        Number of neurons in the layer.
    input_shape: tuple
        Shape of the input to the layer (for tabular data, (n_features,)).
    """

    def __init__(self, n_units, input_shape=None):
        super().__init__()
        self.n_units = n_units
        self._input_shape = input_shape

        # estimated parameters
        self.input = None
        self.output = None
        self.weights = None
        self.biases = None

        self.w_optimizer = None
        self.b_optimizer = None

    def initialize(self, optimizer):
        """
        Initializes the weights and biases, and sets the optimizer.

        Parameters
        ----------
        optimizer: Optimizer
            The optimizer used to update the weights and biases.

        Returns
        -------
        self: DenseLayer
        """
        # uniform random values in [-0.5, 0.5]
        self.weights = np.random.rand(self.input_shape()[0], self.n_units) - 0.5
        self.biases = np.zeros((1, self.n_units))

        # one independent optimizer for weights and one for biases
        self.w_optimizer = copy.deepcopy(optimizer)
        self.b_optimizer = copy.deepcopy(optimizer)

        return self

    def parameters(self):
        """
        Returns the number of parameters of the layer (weights + biases).
        """
        return np.prod(self.weights.shape) + np.prod(self.biases.shape)

    def forward_propagation(self, input, training):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.biases
        return self.output

    def backward_propagation(self, output_error):
        # gradients w.r.t. weights and biases
        weights_error = np.dot(self.input.T, output_error)
        biases_error = np.sum(output_error, axis=0, keepdims=True)

        # error to pass to the previous layer
        input_error = np.dot(output_error, self.weights.T)

        # update the parameters with the optimizers
        self.weights = self.w_optimizer.update(self.weights, weights_error)
        self.biases = self.b_optimizer.update(self.biases, biases_error)

        return input_error

    def output_shape(self):
        """
        Returns the shape of the output of the layer.
        """
        return (self.n_units,)

class Dropout(Layer):
    """
    Randomly sets a fraction of inputs to 0 during training (regularization).

    Parameters
    ----------
    probability: float
        Dropout rate, between 0 and 1.
    """

    def __init__(self, probability):
        super().__init__()
        self.probability = probability

        # estimated parameters
        self.mask = None
        self.input = None
        self.output = None

    def forward_propagation(self, input, training=True):
        if training:
            scaling_factor = 1 / (1 - self.probability)
            self.mask = np.random.binomial(1, 1 - self.probability, size=input.shape)
            self.input = input
            self.output = input * self.mask * scaling_factor
            return self.output
        return input

    def backward_propagation(self, output_error):
        return output_error * self.mask

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0