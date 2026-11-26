
"""K-Means clustering for identifying similar PSL signs."""

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class SignClusterAnalyzer:

    def __init__(self, n_clusters=5, random_state=42):
        if n_clusters < 1:
            raise ValueError("Number of clusters must be positive.")

        self.n_clusters = n_clusters
        self.random_state = random_state

        self.scaler = StandardScaler()
        self.model = None

        self.sign_names = []
        self.sign_vectors = None
        self.cluster_labels = None

    def fit(self, X, y):
        """
        Aggregate samples by sign, then cluster the sign-level vectors.

        X: numeric feature matrix
        y: corresponding sign labels
        """

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional feature matrix.")

        if len(X) != len(y):
            raise ValueError("X and y must have equal lengths.")

        if len(X) == 0:
            raise ValueError("Dataset cannot be empty.")

        if not np.isfinite(X).all():
            raise ValueError("Features contain NaN or infinite values.")

        self.sign_names = list(dict.fromkeys(y.tolist()))

        if len(self.sign_names) < self.n_clusters:
            raise ValueError(
                "Number of unique signs must be at least n_clusters."
            )

        self.sign_vectors = np.vstack([
            X[y == sign].mean(axis=0)
            for sign in self.sign_names
        ])

        scaled_vectors = self.scaler.fit_transform(self.sign_vectors)

        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )

        self.cluster_labels = self.model.fit_predict(scaled_vectors)

        return self

    def get_clusters(self):
        """Return signs grouped by their assigned cluster."""

        if self.model is None:
            raise RuntimeError("Call fit() before requesting clusters.")

        clusters = {}

        for sign, cluster in zip(
            self.sign_names,
            self.cluster_labels
        ):
            clusters.setdefault(int(cluster), []).append(sign)

        return clusters

    def get_similar_signs(self, top_n=10):
        """
        Find pairs of signs with the smallest standardized
        distances between their average feature vectors.
        """

        if self.model is None:
            raise RuntimeError("Call fit() first.")

        if not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")

        scaled = self.scaler.transform(self.sign_vectors)

        distances = pairwise_distances(
            scaled,
            metric="euclidean"
        )

        pairs = []

        for i in range(len(self.sign_names)):
            for j in range(i + 1, len(self.sign_names)):
                pairs.append({
                    "sign_1": self.sign_names[i],
                    "sign_2": self.sign_names[j],
                    "distance": float(distances[i, j]),
                    "same_cluster": bool(
                        self.cluster_labels[i] == self.cluster_labels[j]
                    )
                })

        pairs.sort(key=lambda item: item["distance"])

        return pairs[:top_n]

    def analyze(self, X, y, top_n=10):
        """Run clustering and return a combined analysis."""

        self.fit(X, y)

        return {
            "clusters": self.get_clusters(),
            "similar_signs": self.get_similar_signs(top_n)
        }
