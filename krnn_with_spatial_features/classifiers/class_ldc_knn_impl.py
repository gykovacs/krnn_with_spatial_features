# %% kNN LDC


import math

import numpy as np

import scipy.stats as st
import scipy.spatial

from sklearn.neighbors import NearestNeighbors

import class_ldc_knn_util as knnldcutil


# %%

#######################
# The proposed method #
#######################


class kNNLocalDensityCorrection:
    """
    The proposed method
    """

    def __init__(self,
                 n_neighbors=1,
                 correction=None,
                 w=0.000001):

        self.n_neighbors = n_neighbors
        self.correction = correction
        self.c_g = 0.1
        self.c_r = 0.1
        self.w = w

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

    def r4(self, z, N):
        neg_samples = []
        pos_samples = []

        for i in N:
            if self.labels[i] == self.negative_label:
                neg_samples.append(self.X[i])
            else:
                pos_samples.append(self.X[i])

        # if len(neg_samples) > 20:
        #    mask= np.random.choice(list(range(len(neg_samples))), 20)
        #    neg_samples= [neg_samples[i] for i in range(len(neg_samples)) if i in mask]
        if len(neg_samples) > 10:
            neg_samples = neg_samples[:10]

        if len(neg_samples) <= 1:
            return 1.0

        dist_m = scipy.spatial.distance_matrix(neg_samples, neg_samples)
        dist_m.sort(axis=1)
        mean_dist = np.mean(dist_m[:, 1])
#        max_dist= np.max(np.max(scipy.spatial.distance_matrix(neg_samples, neg_samples)))
        radius = np.linalg.norm(z - neg_samples[-1])

        dist_m2 = scipy.spatial.distance_matrix(z.reshape(1, -1), neg_samples)
        mean_dist2 = np.mean(dist_m2)

        if abs(mean_dist2) < 0.01:
            return 0.0

        # if radius == 0:
        #    return 1.0

        # if mean_dist > 2*radius:
        #    print(mean_dist, 2*radius)

        # return 1.0 - mean_dist/(2*radius)
        return 1.0 - mean_dist/mean_dist2

    def r3(self, z, N):
        neg_samples = []
        pos_samples = []

        for i in N:
            if self.labels[i] == self.negative_label:
                neg_samples.append(self.X[i])
            else:
                pos_samples.append(self.X[i])

        if len(neg_samples) == 0:
            return 1.0

        return 1.0 - knnldcutil.get_weight(z, np.vstack(pos_samples), np.vstack(neg_samples))

    def r1(self, z, N):
        mean_neg = 0.0
        max_dist = 0.0
        num_neg = 0
        num_pos = 0
        min_neg = 100.0
        min_pos = 100.0

        for i in N:
            d = np.linalg.norm(z - self.X[i])
            if self.labels[i] == self.negative_label:
                mean_neg = mean_neg + d
                num_neg = num_neg + 1
                if d < min_neg:
                    min_neg = d
            else:
                num_pos = num_pos + 1
                if d < min_pos:
                    min_pos = d

            if d > max_dist:
                max_dist = d

        if num_neg > 0 and min_pos + min_neg > 0:
            return (1.0 - min_pos/(min_pos + min_neg))*2
        else:
            return 1.0

    def r2(self, z, N):
        mean_neg = 0.0
        mean_pos = 0.0
        max_dist = 0.0
        num_neg = 0
        num_pos = 0
        min_neg = 100.0
        min_pos = 100.0
        min_neg_v = None
        min_pos_v = None

        for i in N:
            d = np.linalg.norm(z - self.X[i])
            if self.labels[i] == self.negative_label:
                mean_neg = mean_neg + d
                num_neg = num_neg + 1
                if d < min_neg:
                    min_neg = d
                    min_neg_v = self.X[i]
            else:
                mean_pos = mean_pos + d
                num_pos = num_pos + 1
                if d < min_pos:
                    min_pos = d
                    min_pos_v = self.X[i]

            if d > max_dist:
                max_dist = d

        if num_neg > 0 and min_pos + min_neg > 0:
            v1 = min_neg_v - z
            v2 = min_pos_v - z

            inner = np.inner(v1, v2)[0][0]
            norm1 = np.inner(v1, v1)[0][0]
            norm2 = np.inner(v2, v2)[0][0]

            if norm1 > 0 and norm2 > 0:
                angle = (1.0 + inner/(norm1*norm2))/2.0
            elif norm1 > 0:
                angle = 0.0
            else:
                angle = 1.0

            prob = mean_neg/(mean_neg + mean_pos)*2

            return prob
        else:
            return 1.0

    def fit(self, X, labels):
        self.X = X
        self.labels = labels

        # Determining the positive class
        self.positive_label = 1
        self.negative_label = 0
        self.classes_ = [self.negative_label, self.positive_label]

        self.num_pos_label = np.sum(labels == self.positive_label)
        self.num_neg_label = np.sum(labels == self.negative_label)

        # if len(self.X) > 3500:
        #    self.nbrs= NearestNeighbors(n_neighbors= len(self.X), n_jobs= 6).fit(self.X)
        # else:
        self.nbrs = NearestNeighbors(n_neighbors=len(self.X)).fit(self.X)
        #self.nbrs= NearestNeighbors(n_neighbors= 50).fit(self.X)

        self.D = float(len(labels))

        self.positive_freq = sum(self.labels == self.positive_label)/self.D
        self.negative_freq = 1 - self.positive_freq

        self.L_g = self.eq1(self.c_g, self.positive_freq, self.D)

    def predict(self, z):
        """
        Predicts the class label of z
        """
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        """
        Predicts the posterior probabilities of class labels
        """

        distances, indices = self.nbrs.kneighbors(z)

        G = list(zip(distances[0], indices[0], self.labels[indices[0]]))

        k_pos, r = 0, 0
        while r < len(G)-1:
            if G[r][2] == self.positive_label:
                k_pos = k_pos + 1
            r = r + 1
            if k_pos >= self.n_neighbors and G[r][2] != G[r-1][2]:
                break

        L_r = self.eq2(self.c_r, float(k_pos)/r, float(r))

        if L_r[0] > self.L_g[1]:
            if r > k_pos:
                lam = (float(k_pos)/float(r - k_pos)) / \
                    (self.positive_freq/self.negative_freq)
            else:
                lam = 1.0
            p_pos = self.eq4(k_pos, r, self.D, lam)
        else:
            p_pos = self.eq3(k_pos, r, self.D)

        if self.correction == None:
            pos_prob = p_pos
        elif self.correction == 'r1':
            pos_prob = max(0.0, min(1, p_pos - self.w/2 +
                                    self.w*self.r1(z, indices[0][:r])))
        elif self.correction == 'r2':
            pos_prob = max(0.0, min(1, p_pos - self.w/2 +
                                    self.w*self.r2(z, indices[0][:r])))
        elif self.correction == 'r3':
            pos_prob = max(0.0, min(1, p_pos - self.w/2 +
                                    self.w*self.r3(z, indices[0][:r])))
        elif self.correction == 'r4':
            pos_prob = max(0.0, min(1, p_pos - self.w/2 +
                                    self.w*self.r4(z, indices[0][:r])))
        elif self.correction == 'pure_r1':
            pos_prob = self.r1(z, indices[0][:r])
        elif self.correction == 'pure_r2':
            pos_prob = self.r2(z, indices[0][:r])
        elif self.correction == 'pure_r3':
            pos_prob = self.r3(z, indices[0][:r])
        elif self.correction == 'pure_r4':
            pos_prob = self.r4(z, indices[0][:r])
        elif self.correction == 'ens_r1':
            pos_prob = 0.5*(p_pos + self.r1(z, indices[0][:r]))
        elif self.correction == 'ens_r2':
            pos_prob = 0.5*(p_pos + self.r2(z, indices[0][:r]))
        elif self.correction == 'ens_r3':
            pos_prob = 0.5*(p_pos + self.r3(z, indices[0][:r]))
        elif self.correction == 'ens_r4':
            pos_prob = 0.5*(p_pos + self.r4(z, indices[0][:r]))
        elif self.correction == 'r_all':
            r1_tmp = self.r1(z, indices[0][:r])
            r3_tmp = self.r3(z, indices[0][:r])
            r4_tmp = self.r4(z, indices[0][:r])
            r_tmp = (r1_tmp + r3_tmp + r4_tmp)/3.0
            pos_prob = max(0.0, min(1, p_pos - self.w/2 + self.w*r_tmp))

        return [[1 - pos_prob, pos_prob]]
