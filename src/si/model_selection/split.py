import numpy as np
from si.data.dataset import Dataset

def train_test_split(dataset, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = dataset.X.shape[0]
    permutations = np.random.permutation(n_samples)

    n_test = int(n_samples * test_size)

    test_idx = permutations[:n_test]
    train_idx = permutations[n_test:]

    train = Dataset(
        X=dataset.X[train_idx],
        y=dataset.y[train_idx] if dataset.y is not None else None,
        features=dataset.features,
        label=dataset.label,
    )
    test = Dataset(
        X=dataset.X[test_idx],
        y=dataset.y[test_idx] if dataset.y is not None else None,
        features=dataset.features,
        label=dataset.label,
    )

    return train, test

