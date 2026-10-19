import numpy as np
from si.base.transformer import Transformer
from si.data.dataset import Dataset
from si.statistics.f_classification import f_classification


class SelectKBest(Transformer):
    """
    Select features according to the k highest F-values.

    Parameters
    ----------
    score_func: callable
        Function that computes F and p values for each feature.
        Defaults to f_classification.
    k: int
        Number of top features to select.
    """

    def __init__(self, score_func=f_classification, k=10, **kwargs):
        super().__init__(**kwargs)
        self.score_func = score_func
        self.k = k

        # estimated parameters
        self.F = None
        self.p = None

    def _fit(self, dataset):
        self.F, self.p = self.score_func(dataset)
        return self

    def _transform(self, dataset):
        idxs = np.argsort(self.F)[-self.k:]
        idxs = np.sort(idxs)

        X_new = dataset.X[:, idxs]

        features_new = None
        if dataset.features is not None:
            features_new = [dataset.features[i] for i in idxs]

        return Dataset(X=X_new, y=dataset.y, features=features_new, label=dataset.label)