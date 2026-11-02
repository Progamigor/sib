import numpy as np
from abc import abstractmethod
from si.neural_networks.layers import Layer


class ActivationLayer(Layer):
    """Base class for activation layers (no learnable parameters)."""

    def forward_propagation(self, input, training):
        self.input = input
        self.output = self.activation_function(self.input)
        return self.output

    def backward_propagation(self, output_error):
        # element-wise: each output depends only on its own input
        return self.derivative(self.input) * output_error

    @abstractmethod
    def activation_function(self, input):
        raise NotImplementedError

    @abstractmethod
    def derivative(self, input):
        raise NotImplementedError

    def output_shape(self):
        return self.input_shape()

    def parameters(self):
        return 0


class SigmoidActivation(ActivationLayer):
    def activation_function(self, input):
        return 1 / (1 + np.exp(-input))

    def derivative(self, input):
        s = self.activation_function(input)
        return s * (1 - s)


class ReLUActivation(ActivationLayer):
    def activation_function(self, input):
        return np.maximum(0, input)

    def derivative(self, input):
        return np.where(input > 0, 1, 0)


class TanhActivation(ActivationLayer):
    def activation_function(self, input):
        return np.tanh(input)