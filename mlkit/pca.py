"""Implementation of PCA."""

import numpy as np

from .base import FeatureTransformer
from .util import validate_data, normalize


class PCA(FeatureTransformer):
    """Perform principal component analysis."""

    def __init__(self, n_dim):
        """Instantiate a PCA transformer.

        Parameters
        ----------
        n_dim : int
            The targeted number of dimensions post-transformation.
        """
        self._n_dim = n_dim
        self.eigen_values = None
        self.eigen_vectors = None

    def fit(self, X, y=None):
        """Fit PCA on a dataset."""
        validate_data(X)
        X = normalize(X)

        cov = np.cov(X.T)
        eig_values, eig_vectors = np.linalg.eig(cov)
        top_indx = self._select_top_features(eig_values, eig_vectors)

        # Select the eigen-vectors and feature vectors from the top indices
        top_eig_vec = -self._reverse_columns(eig_vectors[:, top_indx])
        top_eig_val = self._reverse_columns(eig_values[top_indx])

        self.eigen_values = top_eig_val
        self.eigen_vectors = top_eig_vec

        return self

    def transform(self, X):
        """Transform the feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.

        Return
        ------
        A transformed feature matrix.
        """
        if self.eigen_vectors is None or self.eigen_values is None:
            raise ValueError('PCA must first be fit on a dataset.')

        validate_data(X)

        return np.dot(normalize(X), self.eigen_vectors)

    @staticmethod
    def _reverse_columns(matrix):
        """Reverse the order of the columns in a matrix."""
        return matrix.T[::-1].T

    def _select_top_features(self, eig_values, eig_vectors):
        """Determine the n principal components to keep.

        Parameters
        ----------
        eig_values : array-like
            The eigenvalues of the covarience matrix.
        eig_vectors : array-like
            The eigenvectors of the covarience matrix.

        Returns
        -------
        Array of indices.
        """
        eig_with_indx = list(enumerate(eig_values))
        # Sort the eigen-values and pluck out the top n_dim of them
        top_eig_val = sorted(eig_with_indx, key=lambda x: x[1])[-self._n_dim:]
        # Get the indices of the top eigen-values
        return [x[0] for x in top_eig_val]
