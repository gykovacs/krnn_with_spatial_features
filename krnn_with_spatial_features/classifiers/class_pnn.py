# %% PNN


import math

import scipy.stats as st

from sklearn.neighbors import NearestNeighbors


# %%


class PNN:
    """
    X. Zhang and Y. Li: A positive-biased Nearest Neighbour Algorithm for
    Imbalanced Classification
    In: Advances in Knowledge Discovery and Data Mining
    pp 293--304
    2013
    """

    def __init__(self, n_neighbors=5, c=0.1):
        self.n_neighbors = n_neighbors
        self.c = c

    def fit(self, X, labels):
        self.X = X
        self.labels = labels

        self.nbrs = NearestNeighbors(n_neighbors=self.n_neighbors).fit(self.X)

        self.positive_label = 1
        self.negative_label = 0
        self.classes_ = [self.negative_label, self.positive_label]

        self.N = len(labels)

    def eq1(self, c, f, N):
        """
        Equation 1 in the cited paper, estimating the false positive error rate
        """
        z = abs(st.norm.ppf(c))

        num = f + z*z/(2.0*N) + z*math.sqrt(f*(1-f)/N + z*z/(4.0*N*N))
        denom = 1 + z*z/N

        return num/denom

    def predict(self, z):
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        glob_f = 1.0 - sum(self.labels == self.positive_label) / \
            float(len(self.labels))
        delta = self.eq1(self.c, glob_f, self.N)

        distances, indices = self.nbrs.kneighbors(z)
        G = list(zip(distances[0], indices[0], self.labels[indices[0]]))

        p_q, n_q, i = 0, 0, 0
        while i < len(G) and p_q < int(math.ceil(self.n_neighbors/2.0)):
            if G[i][2] == self.positive_label:
                p_q = p_q + 1
            else:
                n_q = n_q + 1
            i = i + 1
        r = n_q + p_q
        e = self.eq1(self.c, float(n_q)/r, r)

        if r > self.n_neighbors and e < delta:
            p_pos = int(math.ceil(self.n_neighbors/2.0))/int(self.n_neighbors)
        else:
            p_pos = float(p_q)/r

        return [[1.0 - p_pos, p_pos]]
