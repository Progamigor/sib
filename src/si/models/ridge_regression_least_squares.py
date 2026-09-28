import numpy as np
from si.base.model import Model
from si.metrics.mse import mse


class RidgeRegressionLeastSquares(Model):
    """
    Ridge regression solved with the closed-form least squares solution.

    Parameters
    ----------
    l2_penalty: float
        L2 regularization parameter.
    scale: bool
        Whether to scale the data or not.
    """

    def __init__(self, l2_penalty=1.0, scale=True, **kwargs):
        super().__init__(**kwargs)
        self.l2_penalty = l2_penalty
        self.scale = scale

        # estimated parameters
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None

    def _fit(self, dataset):
        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # add a column of 1s for the intercept
        X = np.c_[np.ones(X.shape[0]), X]

        # penalty matrix (intercept is not penalized)
        penalty_matrix = self.l2_penalty * np.eye(X.shape[1])
        penalty_matrix[0, 0] = 0

        # closed-form solution: (XᵀX + λI)⁻¹ Xᵀy
        thetas = np.linalg.solve(X.T @ X + penalty_matrix, X.T @ dataset.y)

        self.theta_zero = thetas[0]
        self.theta = thetas[1:]
        return self

    def _predict(self, dataset):
        X = (dataset.X - self.mean) / self.std if self.scale else dataset.X
        return np.dot(X, self.theta) + self.theta_zero

    def _score(self, dataset, predictions):
        return mse(dataset.y, predictions)