import numpy as np
from si.base.model import Model
from si.metrics.mse import mse


class NeuralNetwork(Model):
    """
    Neural network model: a sequence of layers trained with an optimizer
    to minimize a loss function.

    Parameters
    ----------
    epochs: int
        Number of epochs to train the network.
    batch_size: int
        Number of samples per batch.
    optimizer: class
        Optimizer class to use (it is instantiated here).
    learning_rate: float
        Learning rate of the optimizer.
    verbose: bool
        Whether to print the loss and the metric at each epoch.
    loss: class
        Loss function class to use (it is instantiated here).
    metric: callable
        Metric function used to evaluate the network.
    kwargs:
        Additional keyword arguments passed to the optimizer.
    """

    def __init__(self, epochs=100, batch_size=128, optimizer=None,
                 learning_rate=0.01, verbose=False, loss=None, metric=mse, **kwargs):
        super().__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer(learning_rate=learning_rate, **kwargs)
        self.verbose = verbose
        self.loss = loss()
        self.metric = metric

        # estimated parameters
        self.layers = []
        self.history = {}

    def add(self, layer):
        """
        Adds a layer to the neural network.

        Parameters
        ----------
        layer: Layer
            The layer to add.

        Returns
        -------
        self: NeuralNetwork
        """
        # input shape comes from the previous layer (the first layer
        # must already have its input_shape defined)
        if self.layers:
            layer.set_input_shape(self.layers[-1].output_shape())

        # only layers with weights (like DenseLayer) have initialize
        if hasattr(layer, 'initialize'):
            layer.initialize(self.optimizer)

        self.layers.append(layer)
        return self