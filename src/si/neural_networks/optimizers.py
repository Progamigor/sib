from abc import ABCMeta, abstractmethod


class Optimizer(metaclass=ABCMeta):
    """
    Base class for optimizers.
    An optimizer adjusts the parameters of a layer during training
    to minimize the loss function.

    Parameters
    ----------
    learning_rate: float
        The learning rate of the optimizer.
    """

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w, grad_loss_w):
        """
        Updates the parameters (weights or biases) of a layer.

        Parameters
        ----------
        w: np.ndarray
            The current parameters of the layer.
        grad_loss_w: np.ndarray
            The gradient of the loss function with respect to the parameters.

        Returns
        -------
        w: np.ndarray
            The updated parameters.
        """
        raise NotImplementedError