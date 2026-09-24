import numpy as np
from si.base.model import Model
from si.metrics.accuracy import accuracy


class KNNClassifier(Model):
    """
    K-Nearest Neighbors classifier.
    Estimates the class for a sample based on the k most similar examples.

    Parameters
    ----------
    k: int
        Number of nearest examples to consider.
    distance: callable
        Function that calculates the distance between a sample and the
        samples in the training dataset.
    """

    def __init__(self, k=5, distance=None, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.distance = distance if distance is not None else self._euclidean_distance

        # estimated parameters
        self.dataset = None

    @staticmethod
    def _euclidean_distance(sample, X_train):
        return np.sqrt(np.sum((X_train - sample) ** 2, axis=1))

    def _fit(self, dataset):
        self.dataset = dataset
        return self

    def _predict(self, dataset):
        predictions = []
        for sample in dataset.X:
            distances = self.distance(sample, self.dataset.X)
            k_nearest_idx = np.argsort(distances)[:self.k]
            k_nearest_labels = self.dataset.y[k_nearest_idx]
            labels, counts = np.unique(k_nearest_labels, return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.array(predictions)

    def _score(self, dataset, predictions):
        return accuracy(dataset.y, predictions)