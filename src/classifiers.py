"""Classifiers: a simple baseline plus KNN, SVM and Random Forest behind one interface."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 42


class BaseClassifier(ABC):
    """Every classifier can be trained with fit() and used with predict()."""

    name = "base"

    @abstractmethod
    def fit(self, X, y):
        """Learn from features X and labels y; return self."""

    @abstractmethod
    def predict(self, X):
        """Return one predicted label per row of X."""


class SklearnModel(BaseClassifier):
    """Shared code for models that wrap a scikit-learn estimator."""

    def __init__(self):
        self.model = self.build()

    @abstractmethod
    def build(self):
        """Create the scikit-learn estimator (each subclass overrides this)."""

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class KNNModel(SklearnModel):
    """k-nearest neighbours: a sample gets the label most common among its k closest samples."""

    name = "KNN"

    def __init__(self, k=5):
        self.k = k
        super().__init__()

    def build(self):
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=self.k))


class SVMModel(SklearnModel):
    """Support vector machine with an RBF kernel (draws a curved boundary with the widest gap)."""

    name = "SVM"

    def __init__(self, C=10.0):
        self.C = C
        super().__init__()

    def build(self):
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=self.C, gamma="scale", random_state=SEED))


class RandomForestModel(SklearnModel):
    """Random forest: many decision trees vote on the label."""

    name = "Random Forest"

    def __init__(self, trees=200):
        self.trees = trees
        super().__init__()

    def build(self):
        return RandomForestClassifier(n_estimators=self.trees, random_state=SEED, n_jobs=-1)


class TemplateBaseline(BaseClassifier):
    """Baseline: average each sign into one template, then pick the nearest template."""

    name = "Baseline"

    def __init__(self):
        self.labels = None
        self.templates = None

    def fit(self, X, y):
        y = np.asarray(y)
        self.labels = np.unique(y)
        self.templates = np.stack([X[y == label].mean(axis=0) for label in self.labels])
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, None, :] - self.templates[None, :, :], axis=2)
        return self.labels[distances.argmin(axis=1)]