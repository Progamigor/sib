import numpy as np


def mse(y_true, y_pred):
    """
    Mean squared error between the real and predicted values.

    Parameters
    ----------
    y_true: np.ndarray
        Real values of y.
    y_pred: np.ndarray
        Predicted values of y.

    Returns
    -------
    error: float
        The mean squared error.
    """
    return np.mean((y_true - y_pred) ** 2)