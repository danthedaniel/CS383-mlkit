"""Implementation of K-Nearest Neighbors Classifier."""

from collections import Counter

import numpy as np

from .base import Estimator
from .util import validate_data, normalize


class KNearest(Estimator):
    """K-Nearest Neighbor Classifier."""

    def __init__(self, k):
        """Initialize KNearest.

        Parameters
        ----------
        k : int
            Number of neighbors to sample.
        """
        self._k = k
        self._X = None
        self._y = None

    def fit(self, X, y=None):
        """Fit KNearest on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        if y is None:
            raise ValueError('y must be provided')

        validate_data(X, y)

        self._X = normalize(X)
        self._y = y

        return self

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
        if self._X is None or self._y is None:
            raise ValueError('KNearest must be fit first.')

        validate_data(X)

        return np.array([
            self._class_from_nearest(sample)
            for sample in iter(normalize(X))
        ])

    def _class_from_nearest(self, sample):
        """Find the most frequent class in the nearest k samples.

        Parameters
        ----------
        sample : array-like {n_features, }
            The data point to measure distances from.

        Returns
        -------
        Most common class for the nearest samples.
        """
        # Calculate distance to every datapoint in the training set
        dists = [
            (indx, np.linalg.norm(sample - x))
            for indx, x in enumerate(iter(self._X))
        ]

        # Find the nearest self._k
        lowest_dists = sorted(dists, key=lambda x: x[1])[:self._k]
        lowest_dists_indx = [x[0] for x in lowest_dists]
        nearest_k = list(self._y[lowest_dists_indx])

        return Counter(nearest_k).most_common(1)[0][0]
