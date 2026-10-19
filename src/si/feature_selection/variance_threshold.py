import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset


class VarianceThreshold(Transformer):
    """
    Feature selection method that removes features with variance
    below a given threshold.

    Parameters
    ----------
    threshold: float
        Variance cutoff value. Features with variance below this
        value are removed.
    """

    def __init__(self, threshold=0.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

        # estimated parameters
        self.variance = None

    def _fit(self, dataset):
        self.variance = np.var(dataset.X, axis=0)
        return self

    def _transform(self, dataset):
        mask = self.variance > self.threshold
        X_new = dataset.X[:, mask]

        features_new = None
        if dataset.features is not None:
            features_new = [f for f, keep in zip(dataset.features, mask) if keep]

        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)