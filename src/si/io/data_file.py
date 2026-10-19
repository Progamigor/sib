import numpy as np
from si.data.dataset import Dataset

def read_data_file(filename, sep=",", label=False):
    raw = np.genfromtxt(filename, delimiter=sep)

    if label:
        X = raw[:, :-1]
        y = raw[:, -1]
    else:
        X = raw
        y = None

    return Dataset(X, y)

def write_data_file(filename, dataset, sep=",", label=False):
    if label and dataset.y is not None:
        data = np.column_stack((dataset.X, dataset.y))
    else:
        data = dataset.X

    np.savetxt(filename, data, delimiter=sep)