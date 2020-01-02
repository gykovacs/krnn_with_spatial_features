# %% Simple kNN


import numpy as np

from sklearn.neighbors import NearestNeighbors


# %%


"""
Simple kNN implementation
"""


class kNNSimple:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, labels):
        self.X = X
        self.labels = labels

        self.positive_label = 1
        self.negative_label = 0
        self.classes_ = [self.negative_label, self.positive_label]

        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X)

    def predict(self, z):
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        distances, indices = self.nbrs.kneighbors(z)
        nearest_labels = self.labels[indices[0]]

        return [[np.sum(nearest_labels == self.negative_label)/float(self.n_neighbors),
                 np.sum(nearest_labels == self.positive_label)/float(self.n_neighbors)]]
