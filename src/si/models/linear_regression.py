import numpy as np
from si.base.model import Model
from si.metrics.mse import mse


class RidgeRegression(Model):
    """
    Linear regression with L2 regularization, trained by gradient descent.

    Parameters
    ----------
    l2_penalty: float
        L2 regularization parameter.
    alpha: float
        Learning rate.
    max_iter: int
        Maximum number of iterations.
    patience: int
        Maximum number of iterations without improvement allowed.
    scale: bool
        Whether to scale the data or not.
    """

    def __init__(self, l2_penalty=1.0, alpha=0.001, max_iter=2000,
                 patience=100, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.alpha = alpha
        self.max_iter = max_iter
        self.patience = patience
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
        self.cost_history = {}

    def _fit(self, dataset):
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        m, n = X.shape
        self.theta = np.zeros(n)
        self.theta_zero = 0.0
        self.cost_history = {}
        early_stopping = 0

        for i in range(self.max_iter):
            y_pred = np.dot(X, self.theta) + self.theta_zero
            error = y_pred - dataset.y

            # gradient descent step (theta is penalized, theta_zero is not)
            gradient = (self.alpha / m) * np.dot(error, X)
            penalization = self.theta * (1 - self.alpha * self.l2_penalty / m)
            self.theta = penalization - gradient
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            # cost after the update
            y_pred = np.dot(X, self.theta) + self.theta_zero
            self.cost_history[i] = self.cost(dataset.y, y_pred)

            # early stopping
            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            if early_stopping >= self.patience:
                break

        return self

    def _predict(self, dataset):
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset, predictions):
        return mse(dataset.y, predictions)

    def cost(self, y_true, y_pred):
        m = len(y_true)
        squared_error = np.sum((y_true - y_pred) ** 2)
        penalty = self.l2_penalty * np.sum(self.theta ** 2)
        return (squared_error + penalty) / (2 * m)