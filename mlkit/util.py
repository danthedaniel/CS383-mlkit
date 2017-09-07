from math import floor

import numpy as np


def validate_data(X, y=None):
    """Check that a feature matrix and class vector are appropriately shaped."""
    assert len(X.shape) == 2
    assert X.shape[0] > 0
    assert X.shape[1] > 0

    if y is not None:
        assert len(y.shape) == 1
        assert X.shape[0] == y.shape[0]


def normalize(*matrices):
    """Normalize all features in each matrix so they have mean 0 and std dev. 1.

    Parameters
    ----------
    matrices : array of matrices
        Numpy matrices.

    Returns
    -------
    A tuple of normalized feature matrices.
    """
    def normalize_matrix(x):
        std = np.std(x, axis=0, ddof=1)
        if np.any(std == 0.0):
            return x

        return (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)

    if len(matrices) > 1 or len(matrices) == 0:
        return tuple([normalize_matrix(x) for x in matrices])
    else:
        return normalize_matrix(matrices[0])


def root_mean_squared_error(y_pred, y_true):
    """Calculated the RMSE of a set of target predictions.

    Parameters
    ----------
    y_pred : array-like {n_samples, }
        Predicted target values.
    y_true : array-like {n_samples, }
        Actual target values.

    Returns
    -------
    RMSE of y_pred against y_true.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Predicted and actual targets are of different shape')

    mse = np.sum((y_pred - y_true) ** 2) / y_pred.shape[0]
    return np.sqrt(mse)


def accuracy(y_pred, y_true):
    """Proportion of correctly classified samples.

    Parameters
    ----------
    y_pred : array-like {n_samples, }
        Predicted target values.
    y_true : array-like {n_samples, }
        Actual target values.

    Returns
    -------
    Accuracy as a proporition of 1.
    """
    if y_pred.shape != y_true.shape:
        raise ValueError('Predicted and actual targets are of different shape')

    return np.sum((y_pred == y_true)) / y_pred.shape[0]


def precision(y_pred, y_true, positive_class=1.0, negative_class=0.0):
    """True positives over true and false positives."""
    true_positives = np.sum((y_pred == positive_class) & (y_true == positive_class))
    false_positives = np.sum((y_pred == positive_class) & (y_true == negative_class))

    return true_positives / (true_positives + false_positives)


def recall(y_pred, y_true, positive_class=1.0, negative_class=0.0):
    """True positives over true positives and false negatives."""
    true_positives = np.sum((y_pred == positive_class) & (y_true == positive_class))
    false_negatives = np.sum((y_pred == negative_class) & (y_true == positive_class))

    return true_positives / (true_positives + false_negatives)


def f_measure(y_pred, y_true, positive_class=1.0, negative_class=0.0):
    """2 * precision * recall / (precision + recall)."""
    precision_val = precision(y_pred, y_true, positive_class, negative_class)
    recall_val = recall(y_pred, y_true, positive_class, negative_class)

    return 2 * precision_val * recall_val / (precision_val + recall_val)


def train_test_split(X, y, train_size=0.67):
    """Break up feature and target matrices into training and testing sets.

    Parameters
    ----------
    X : array-like {n_samples, n_features}
        Feature matrix.
    y : array-like {n_samples, }
        Target vector.
    train_size : float
        Proportion of data to use as training data.

    Returns
    -------
    4-tuple of values:
        X_train,
        y_train,
        X_test,
        y_test
    """
    validate_data(X, y)

    n_train = floor(X.shape[0] * train_size)
    n_test = X.shape[0] - n_train

    return (
        X[0:n_train, :],  # X train
        y[0:n_train],     # y train
        X[-n_test:, :],   # X test
        y[-n_test:]       # y test
    )


def k_fold_cross_validation(X, y, k, est, scorer=root_mean_squared_error):
    """Perform k-fold cross validation on an estimator.

    Parameters
    ----------
    X : array-like {n_samples, n_features}
        Feature matrix.
    y : array-like {n_samples, }
        Target vector.
    k : int
        Number of times to perform CV.
    est : Estimator
        Estimator to test.
    scorer : Callable
        Function to score predicted labels.

    Returns
    -------
    list of size k with scores from each CV.
    """
    validate_data(X, y)

    scores = []
    train_size = floor(1 / k * X.shape[0])

    if train_size == 0:
        raise ValueError('k is too large for the given dataset')

    for _ in range(k):
        X_train, y_train, X_test, y_test = train_test_split(X, y, 1 / k)

        est.fit(X_train, y_train)
        scores.append(est.score(X_test, y_test, scorer))

        # Roll X and y by train_size so that the next test will be on
        # different data.
        X = np.roll(X, train_size, axis=0)
        y = np.roll(y, train_size, axis=0)

    return scores
