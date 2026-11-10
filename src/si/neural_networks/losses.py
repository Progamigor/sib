from abc import ABCMeta, abstractmethod


class LossFunction(metaclass=ABCMeta):
    """
    Base class for loss functions.
    Defines the interface that all loss functions must implement.
    """

    @abstractmethod
    def loss(self, y_true, y_pred):
        """
        Computes the loss between the true and predicted values.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels.
        y_pred: np.ndarray
            The predicted labels.

        Returns
        -------
        loss: float
            The loss value.
        """
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_true, y_pred):
        """
        Computes the derivative of the loss with respect to the predictions.

        Parameters
        ----------
        y_true: np.ndarray
            The true labels.
        y_pred: np.ndarray
            The predicted labels.

        Returns
        -------
        derivative: np.ndarray
            The derivative of the loss.
        """
        raise NotImplementedError