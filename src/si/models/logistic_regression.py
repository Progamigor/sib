import numpy as np
from si.base.model import Model
from si.metrics.accuracy import accuracy
from si.statistics.sigmoid_function import sigmoid_function


class LogisticRegression(Model):
    """
    Logistic regression with L2 regularization, trained by gradient descent.

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
            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
            error = y_pred - dataset.y

            gradient = (self.alpha / m) * np.dot(error, X)
            penalization = self.theta * (1 - self.alpha * self.l2_penalty / m)
            self.theta = penalization - gradient
            self.theta_zero = self.theta_zero - (self.alpha / m) * np.sum(error)

            y_pred = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
            self.cost_history[i] = self.cost(dataset.y, y_pred)

            if i > 0 and self.cost_history[i] >= self.cost_history[i - 1]:
                early_stopping += 1
            else:
                early_stopping = 0
            if early_stopping >= self.patience:
                break

        return self

    def _predict(self, dataset):
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        probabilities = sigmoid_function(np.dot(X, self.theta) + self.theta_zero)
        return np.where(probabilities >= 0.5, 1, 0)

    def _score(self, dataset, predictions):
        return accuracy(dataset.y, predictions)

    def cost(self, y_true, y_pred):
        m = len(y_true)
        cost = (-y_true * np.log(y_pred)) - (1 - y_true) * np.log(1 - y_pred)
        cost = np.sum(cost) / m
        cost = cost + (self.l2_penalty / (2 * m)) * np.sum(self.theta ** 2)
        return cost