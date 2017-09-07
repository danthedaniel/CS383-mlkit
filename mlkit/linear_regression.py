""""Implementation of a Linear Regression Estimator."""

import numpy as np

from .base import Estimator
from .util import validate_data, normalize


class LinearRegression(Estimator):
    """Perform Linear Regression."""

    def __init__(self):
        """Initialize LinearRegression."""
        self._coeff = None  # Coefficients for each feature
        self._mean = None
        self._std = None

    def fit(self, X, y):
        """Fit LinearRegression on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        validate_data(X, y)
        self._mean = np.mean(X, axis=0)
        self._std = np.std(X, axis=0)
        X = self._add_ones(normalize(X))

        # Coefficients can be found from: (X^T . X)^-1 . (X^T . y)
        self._coeff = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

        return self

    @staticmethod
    def _add_ones(X):
        """Add a column of ones to the beginning of a matrix."""
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def predict(self, X):
        """Predict labels for the feature matrix based on the training data.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.

        Return
        ------
        Predicted targets.
        """
        if self._coeff is None or self._mean is None or self._std is None:
            raise ValueError('LinearRegression must be fit first.')

        validate_data(X)
        X = (X - self._mean) / self._std
        X = self._add_ones(X)

        return np.dot(X, self._coeff)
