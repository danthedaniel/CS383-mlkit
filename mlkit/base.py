"""Base Classes for ML Implementations."""

from abc import ABCMeta, abstractmethod


class FeatureTransformer(object):
    """Base class for feature transformers."""

    __metaclass__ = ABCMeta

    def fit(self, X, y=None):
        """Fit the feature transformer on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        return self

    @abstractmethod
    def transform(self, X):
        """Transform the feature matrix based on the training data.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.

        Return
        ------
        A transformed feature matrix.
        """
        pass

    def fit_transform(self, X, y=None):
        """Convenience method for calling fit and transform successively.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.

        Return
        ------
        A transformed feature matrix.
        """
        return self.fit(X, y).transform(X)


class Estimator(object):
    """Base class for estimators."""

    __metaclass__ = ABCMeta

    def fit(self, X, y=None):
        """Fit the estimator on a feature matrix.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.
        """
        return self

    @abstractmethod
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
        pass

    def score(self, X, y, scorer):
        """Score the estimator based on testing data."""
        return scorer(self.predict(X), y)

    def fit_predict(self, X, y=None):
        """Convenience method for calling fit and predict successively.

        Parameters
        ----------
        X : array-like {n_samples, n_features}
            A feature matrix.
        y : array-like {n_samples, }
            A target vector.

        Return
        ------
        Predicted targets.
        """
        return self.fit(X, y).predict(X)
