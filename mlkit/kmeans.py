"""Implementation of K-Means Clustering."""

import numpy as np

from .base import Estimator
from .util import validate_data, normalize


class KMeans(Estimator):
    """Perform K-Means Clustering."""

    def __init__(self, k=2, epsilon=(2 ** -23), seed=42):
        """Initialize KMeans.

        Parameters
        ----------
        k : int
            Number of clusters to find.
        epsilon : float
            Tolerance for cluster center changes between iterations. Once
            centers move less than this, model fitting ceases.
        seed : int
            Random number generator seed.
        """
        self._k = k
        self._epsilon = epsilon
        self._seed = seed
        self._centers = None
        self._iterations = None

    def fit(self, X, y=None):
        """Fit KMeans on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        np.random.seed(self._seed)
        validate_data(X)
        X = normalize(X)

        init_centers_indx = np.random.choice(list(range(X.shape[0])), self._k)
        init_centers = X[init_centers_indx, :]
        self._centers = self._find_centers(X, init_centers)

        return self

    def _find_centers(self, X, centers, iteration=1):
        """Determine the k centers for the dataset, recursively.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            Feature matrix.
        centers : array-like {k, }
            Centers of each cluster.
        iteration : int
            Current iteration count.

        Returns
        -------
        Centers of each cluster.
        """
        # Mark each sample as belonging to a cluster
        labels = np.array([
            self._closest_center(sample, centers)
            for sample in iter(X)
        ])
        # Find new centers from the samples' labels
        new_centers = np.array([
            np.mean(X[labels == label], axis=0)
            for label in range(self._k)
        ])
        # Check if the centers have moved significantly from the last iteration
        center_deltas = np.linalg.norm(new_centers - centers, axis=0)

        if all(center_deltas < self._epsilon):
            self._iterations = iteration
            return new_centers
        else:
            return self._find_centers(X, new_centers, iteration + 1)

    @staticmethod
    def _closest_center(sample, centers):
        """Find the center that is closest to a sample from the dataset.

        Parameters
        ----------
        sample : array-like {n_features, }
            A single sample from the dataset.
        centers : array-like {k, }
            The current k centers of each cluster.

        Returns
        -------
        Index of the closest center.
        """
        # Find distance from the sample to each center
        distances = [np.linalg.norm(sample - center) for center in centers]
        return distances.index(min(distances))  # Pick the closest center

    def predict(self, X):
        """Predict labels for the feature matrix based on the training data.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.

        Return
        ------
        Predicted classes.
        """
        if self._centers is None:
            raise ValueError('KMeans must be fit first.')

        validate_data(X)
        X = normalize(X)

        return np.array([
            self._closest_center(sample, self._centers)
            for sample in iter(X)
        ])
