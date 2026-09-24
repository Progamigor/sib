import pandas as pd
from si.data.dataset import Dataset

def read_csv(filename, sep=",", features=False, label=False):
    df = pd.read_csv(filename, sep=sep, header=0 if features else None)

    if label:
        X = df.iloc[:, :-1].to_numpy()
        y = df.iloc[:, -1].to_numpy()
        feature_names = list(df.columns[:-1]) if features else None
        label_name = df.columns[-1] if features else None
    else:
        X = df.to_numpy()
        y = None
        feature_names = list(df.columns) if features else None
        label_name = None

    return Dataset(X, y, features=feature_names, label=label_name)