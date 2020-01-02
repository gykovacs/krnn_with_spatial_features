# %% ENN


import collections
import math

import numpy as np

import scipy.stats as st

from sklearn.neighbors import NearestNeighbors


# %%


class ENN:
    """
    Li and Zhang: Improving k nearest neighbor with exemplar generalization for imbalanced classification
    """

    def __init__(self, n_neighbors=5, c=0.1):
        self.n_neighbors = n_neighbors
        self.c = c

    def fit(self, X, labels):
        self.X = X
        self.labels = labels
        self.nbrs = NearestNeighbors(n_neighbors=len(X)).fit(self.X)

        freqs = collections.Counter(labels).most_common()
        self.positive_label = freqs[-1][0]
        self.negative_label = freqs[-2][0]
        self.classes_ = [self.negative_label, self.positive_label]

        self.N = float(len(labels))
        self.positive_freq = sum(self.labels == self.positive_label)/self.N
        self.negative_freq = 1 - self.positive_freq
        self.determine_positive_pivots()

    def eq1(self, c, f, N):
        """
        Equation 1 in the cited paper estimating the false positive error rate
        """
        z = abs(st.norm.ppf(c/2.0))
        num = f + z*z/(2.0*N) + z*math.sqrt(f*(1 - f)/N + z*z/(4.0*N*N))
        denom = 1.0 + z*z/N
        return num/denom

    def determine_positive_pivots(self):
        self.P = {}
        delta = self.eq1(self.c, self.negative_freq, self.N)
        for i in range(len(self.X)):
            distances, indices = self.nbrs.kneighbors(self.X[i].reshape(1, -1))
            G = list(zip(distances[0][1:], indices[0]
                         [1:], self.labels[indices[0][1:]]))

            for k in range(1, len(G)):
                if G[k][2] == self.positive_label:
                    break

            r = np.linalg.norm(self.X[i] - self.X[G[k][1]])
            f = float(k - 1)/float(k + 1)
            p = self.eq1(self.c, f, k)
            if p <= delta:
                self.P[i] = r

    def predict(self, z):
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        distances, indices = self.nbrs.kneighbors(z)
        for i in indices[0]:
            if i in self.P:
                distances[0][i] = distances[0][i] - self.P[i]

        G = sorted(zip(distances[0], indices[0],
                       self.labels[indices[0]]), key=lambda x: x[0])

        nearest_labels = [g[2] for g in G[:self.n_neighbors]]

        neg_num = np.sum(nearest_labels == self.negative_label)
        pos_num = np.sum(nearest_labels == self.positive_label)

        return [[neg_num / float(neg_num + pos_num), pos_num / float(neg_num + pos_num)]]
