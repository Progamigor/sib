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

if __name__ == "__main__":
    import numpy as np

    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([0, 1, 0, 1, 0])
    dataset = Dataset(X, y, features=["f1", "f2"], label="target")

    train, test = train_test_split(dataset, test_size=0.2, random_state=42)
    print(train.shape())
    print(test.shape())