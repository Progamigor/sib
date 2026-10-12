
def sigmoid_function(X):
    """
    Computes the sigmoid function for the given input values.

    Parameters
    ----------
    X: np.ndarray
        Input values.

    Returns
    -------
    probabilities: np.ndarray
        The probability of each value being 1.
    """
    return 1 / (1 + np.exp(-X))