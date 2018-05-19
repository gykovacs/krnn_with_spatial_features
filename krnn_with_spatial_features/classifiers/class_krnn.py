# %% kRNN


import collections
import math

import scipy.stats as st

from sklearn.neighbors import NearestNeighbors


# %%


"""
Zhang et al: KRNN: k Rare class Nearest Neighbour classification
"""


class KRNN:
    def __init__(self, n_neighbors=5, c_g=0.1, c_r=0.1):
        self.n_neighbors = n_neighbors
        self.c_g = c_g
        self.c_r = c_r

    def fit(self, X, labels):
        self.X = X
        self.labels = labels

        self.nbrs = NearestNeighbors(n_neighbors=len(X), n_jobs=6).fit(self.X)

        freqs = collections.Counter(labels).most_common()
        self.positive_label = freqs[-1][0]
        self.negative_label = freqs[-2][0]
        self.classes_ = [self.negative_label, self.positive_label]

        self.D = float(len(labels))

        self.positive_freq = sum(self.labels == self.positive_label)/self.D
        self.negative_freq = 1 - self.positive_freq

        self.L_g = self.eq1(self.c_g, self.positive_freq, self.D)

    def eq1(self, c, q, r):
        """
        Equation 1 in the cited paper estimating the global confidence interval
        """

        z = abs(st.norm.ppf(c/2.0))
        return (q - z*math.sqrt(q*(1 - q)/r), q + z*math.sqrt(q*(1 - q)/r))

    def eq2(self, c, q, r):
        """
        Equation 2 in the cited paper estimating the local confidence interval
        """

        z = abs(st.norm.ppf(c/2.0))
        pm = z*math.sqrt(q*(1 - q)/r + z*z/(4.0*r*r))
        denom = 1.0 + z*z/r
        return ((q + z*z/2.0*r - pm)/denom, (q + z*z/2.0*r + pm)/denom)

    def eq3(self, k_q, n, D):
        """
        Equation 3 in the cited paper estimating the positive posterior probability
        """

        return (k_q + 1.0/D)/(k_q + n + 2.0/D)

    def eq4(self, k_q, n, D, lam):
        """
        Equation 4 in the cited paper estimating the positive posterior probability
        """

        return (k_q + 1.0/D)/(k_q + 1.0/lam*n + 2.0/D)

    def predict(self, z):
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        """
        Determining the posterior probabilities
        """
        distances, indices = self.nbrs.kneighbors(z)
        G = list(zip(distances[0], indices[0], self.labels[indices[0]]))

        k_q, i = 0, 0
        while i < len(G)-1 and (k_q < self.n_neighbors or G[i][2] == G[i+1][2]):
            if G[i][2] == self.positive_label:
                k_q = k_q + 1
            i = i + 1
        r = i

        if k_q == r:
            return [[0.0, 1.0]]

        L_r = self.eq2(self.c_r, float(k_q)/r, float(r))

        # if L_r[0] > self.L_g[1]:
        lam = (float(k_q)/float(r - k_q)) / \
            (self.positive_freq/self.negative_freq)
        p_pos = self.eq4(k_q, r, self.D, lam)
        # else:
        #p_pos= self.eq3(k_q, r, self.D)

        return [[1.0 - p_pos, p_pos]]
