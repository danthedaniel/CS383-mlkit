"""Implementation of an Artificial Neural Network."""

import math

import numpy as np

from .base import Estimator
from .util import validate_data, normalize


def sigmoid(x, derivative=False):
    """Sigmoid activation function."""
    if derivative:
        return x * (1 - x)

    return 1 / (1 + math.e ** (-x))


class NeuralNet(Estimator):
    """An artificial neural network classifier with a single hidden layer."""

    def __init__(self, hidden_size=20, learning_rate=0.5, activation=sigmoid,
                 seed=0, training_iterations=1000):
        """Instantiate a NeauralNet.

        Parameters
        ----------
        hidden_size : int
            Number of neurons in the hidden layer.
        learning_rate : float
            Learning rate for batch gradient descent.
        activation : callable
            Activation function.
        seed : int
            Random state seed.
        training_iterations : int
            Number of iterations to perform in fit().
        """
        self._hidden_size = 20
        self._learning_rate = learning_rate
        self._activation = activation
        self._seed = seed
        self._training_iterations = training_iterations

        self._synapses = None
        self._hidden_layer = None
        self._classes = None

    def fit(self, X, y=None):
        """Fit the estimator on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        validate_data(X, y)
        X = self._add_bias_feature(normalize(X))
        self._classes = np.unique(y)
        np.random.seed(self._seed)

        n_samples, n_features = X.shape
        self._synapses = 2 * np.random.random((n_features, )) - 1

        for _ in range(self._training_iterations):
            hidden_layer_delta = self._forward_propogation(X, y)
            self._backward_propogation(X, hidden_layer_delta)

        self._training_predictions = self._activation(np.dot(X, self._synapses))

        return self

    @staticmethod
    def _add_bias_feature(X):
        bias = np.ones((X.shape[0], 1))
        return np.hstack([X, bias])

    def _forward_propogation(self, X, y):
        hidden_layer = self._activation(np.dot(X, self._synapses))
        error = y - hidden_layer
        return error * self._activation(hidden_layer, True)

    def _backward_propogation(self, X, hidden_layer_delta):
        self._synapses += self._learning_rate * np.dot(X.T, hidden_layer_delta)

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
        validate_data(X)
        X = self._add_bias_feature(normalize(X))

        if self._synapses is None:
            raise ValueError('NeuralNet has not been fit.')

        return self._activation(np.dot(X, self._synapses))
