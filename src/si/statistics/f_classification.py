from scipy.stats import f_oneway


def f_classification(dataset):
    """
    Scoring function for classification problems. Computes one-way ANOVA
    F-value for each feature, testing whether the feature's mean differs
    significantly across the dataset's classes.

    Parameters
    ----------
    dataset: Dataset
        The dataset to analyze.

    Returns
    -------
    F: tuple
        F values for each feature.
    p: tuple
        p-values for each feature.
    """
    classes = dataset.get_classes()

    groups = [dataset.X[dataset.y == c] for c in classes]

    F, p = f_oneway(*groups)

    return F, p