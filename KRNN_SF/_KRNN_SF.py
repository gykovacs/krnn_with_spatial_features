"""
Created on Sat Feb 17 11:02:24 2018

@author: gykovacs
"""

__all__= ['KRNN_SF']

import math

import numpy as np
import pandas as pd

import scipy.stats as st
import scipy.spatial

from sklearn.neighbors import NearestNeighbors
    
def L2_norm(x):
    return np.sqrt(x.T.dot(x))

def pair_likelihood_pdf(query, pos_sample, neg_sample):
    # Returns a normal pdf
    # The weight of the pdf is closer to zero the more positive the query is

    t = query.reshape(-1, 1)
    xi = pos_sample.reshape(-1, 1)
    xj = neg_sample.reshape(-1, 1)

    yi = xi - t
    xij = xi - xj

    # x_ij length
    d = L2_norm(xij)

    yi_ = xi.T.dot(xij) / d

    # Distance from x_ij
    l = np.sqrt(yi.T.dot(yi) - yi_.T.dot(yi_))

    # Scaled down by d
    mu = yi_ / d

    alpha = 1.0
    scale = alpha * np.sqrt(l/d)

    return st.norm(mu, scale)


def get_point_membership_distribution(query, pos_samples, neg_samples):
    return [
        pair_likelihood_pdf(query,
                            pos_samples[pos_sample_inx, :].T,
                            neg_samples[neg_sample_inx, :].T)
        for neg_sample_inx in np.arange(neg_samples.shape[0])
        for pos_sample_inx in np.arange(pos_samples.shape[0])
    ]


def get_weight(t, pos_samples, neg_samples):
    if len(neg_samples) > 5:
        neg_samples = neg_samples[:5]

    point_membership_distribution =\
        get_point_membership_distribution(t, pos_samples, neg_samples)

    # return np.mean([max(0, min(1, d)) for d in point_membership_distribution])
    lower_bound = min([pdf.ppf(0.01)
                       for pdf in point_membership_distribution])[0]
    upper_bound = max([pdf.ppf(0.99)
                       for pdf in point_membership_distribution])[0]

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.5

    def eval_pdfs(x):
        return np.sum([pdf.pdf(x) for pdf in point_membership_distribution])

    xs = np.linspace(lower_bound, upper_bound, num=100)
    ys = np.array([eval_pdfs(x) for x in xs])
    ys_pdf = ys / np.sum(ys)
    ys_cdf = np.cumsum(ys_pdf)

    def get_percentile(xs, ys_cdf, percent):
        mask = ys_cdf < percent
        tmp = np.where(mask)[0]
        if len(tmp) == 0:
            return 0.5
        last_true = tmp[-1]
        return xs[last_true]

    return max(0.0, min(1.0, get_percentile(xs, ys_cdf, 0.5)))

class KRNN_SF:
    """
    The implementation of kRNN improved by point pattern ranking based on
    spatial features.
    
    The original kRNN technique is described in
        Zhang et al (2017): kRNN: k Rare class Nearest Neighbour classification, 
        Pattern Recognition, 62, p. 33-44
        
    This technique is described in details in
        László et al (2018): Improving the Performance of the k Rare Class 
        Nearest Neighbor Classifier by the Ranking of Point Patterns. 
        In: Lecture Notes in Computer Science, vol 10833, p. 265-283
        doi: https://doi.org/10.1007/978-3-319-90050-6_15
        link: https://link.springer.com/chapter/10.1007/978-3-319-90050-6_15
        preprint: https://drive.google.com/open?id=1aQct5L6DgYvRnE6wDMKzekhZvxEyktGz
    """
    def __init__(self, 
                 n_neighbors= 1, 
                 correction= None,
                 w= 0.000001,
                 c_g= 0.1,
                 c_r= 0.1,
                 cutoff= 50,
                 n_jobs= -1):
        """
        Args:
            n_neighbors (int): number of neighbors to use
            correction (str): None for pure kRNN, 
                                'r1' for ranking function 1, 
                                'r2' for ranking function 2, 
                                'r3' for ranking function 3,
                                'r_all' for the ensemble of three ranking functions
                                'r1_pure' for using ranking function 1 as the positive probability estimator
                                'r2_pure' for using ranking function 1 as the positive probability estimator
                                'r3_pure' for using ranking function 1 as the positive probability estimator
            w (float): bandwidth
            c_g (float): global confidence level
            c_r (float): local confidence level
            cutoff (int): cutting off the queried number of neighbors as this number,
                            use None for no cutoff
            n_jobs (int): number of processes to use in the nearest neighbors search
        """
        self.n_neighbors= n_neighbors
        self.correction= correction
        self.w= w
        self.c_g= c_g
        self.c_r= c_r
        self.cutoff= cutoff
    
    def eq1(self, c, q, r):
        """
        Equation 1 in the original kRNN paper estimating the global confidence interval
        """
        
        z= abs(st.norm.ppf(c/2.0))
        return (q - z*math.sqrt(q*(1 - q)/r), q + z*math.sqrt(q*(1 - q)/r))
        
    def eq2(self, c, q, r):
        """
        Equation 2 in the original kRNN paper estimating the local confidence interval
        """
        
        z= abs(st.norm.ppf(c/2.0))
        pm= z*math.sqrt(q*(1 - q)/r + z*z/(4.0*r*r))
        denom= 1.0 + z*z/r
        return ((q + z*z/2.0*r - pm)/denom, (q + z*z/2.0*r + pm)/denom)
    
    def eq3(self, k_q, n, D):
        """
        Equation 3 in the original kRNN paper estimating the positive posterior probability
        """
        
        return (k_q + 1.0/D)/(k_q + n + 2.0/D)
        
    def eq4(self, k_q, n, D, lam):
        """
        Equation 4 in the original kRNN paper estimating the positive posterior probability
        """
        
        return (k_q + 1.0/D)/(k_q + 1.0/lam*n + 2.0/D)
    
    def r1(self, z, N):
        """
        Ranking function #1 in the paper describing the proposed method.
        Args:
            z (np.array): the query point
            N (np.array): the query neighborhood
        """
        mean_neg= 0.0
        max_dist= 0.0
        num_neg= 0
        num_pos= 0
        min_neg= 100.0
        min_pos= 100.0
        
        for i in N:
            d= np.linalg.norm(z - self.X[i])
            if self.labels[i] == self.negative_label:
                mean_neg= mean_neg + d
                num_neg= num_neg + 1
                if d < min_neg:
                    min_neg= d
            else:
                num_pos= num_pos + 1
                if d < min_pos:
                    min_pos= d
        
            if d > max_dist:
                max_dist= d
        
        if num_neg > 0 and min_pos + min_neg > 0:
            return (1.0 - min_pos/(min_pos + min_neg))*2
        else:
            return 1.0
    
    def r2(self, z, N):
        """
        Ranking function #2 in the paper describing the proposed method.
        Args:
            z (np.array): the query point
            N (np.array): the query neighborhood
        """
        
        neg_samples= []
        pos_samples= []
        
        for i in N:
            if self.labels[i] == self.negative_label:
                neg_samples.append(self.X[i])
            else:
                pos_samples.append(self.X[i])
        
        if len(neg_samples) == 0:
            return 1.0
        
        return 1.0 - get_weight(z, np.vstack(pos_samples), np.vstack(neg_samples))
    
    def r3(self, z, N):
        """
        Ranking function #3 in the paper describing the proposed method.
        Args:
            z (np.array): the query point
            N (np.array): the query neighborhood
        """
        
        neg_samples= []
        pos_samples= []
        
        for i in N:
            if self.labels[i] == self.negative_label:
                neg_samples.append(self.X[i])
            else:
                pos_samples.append(self.X[i])
    
        if len(neg_samples) > 10:
            neg_samples= neg_samples[:10]
        
        if len(neg_samples) <= 1:
            return 1.0
        
        dist_m= scipy.spatial.distance_matrix(neg_samples, neg_samples)
        dist_m.sort(axis= 1)
        mean_dist= np.mean(dist_m[:,1])
        
        dist_m2= scipy.spatial.distance_matrix(z.reshape(1,-1), neg_samples)
        mean_dist2= np.mean(dist_m2)
        
        if abs(mean_dist2) < 0.01:
            return 0.0
    
        return 1.0 - mean_dist/mean_dist2
  
    def fit(self, X, labels):
        """
        Fit function following the usual sklearn interface
        Args:
            X (matrix): the independent variables
            labels (array): the dependent variable
        """
        self.X= X
        self.labels= labels
        
        # determining the positive class
        self.positive_label= 1
        self.negative_label= 0
        self.classes_= [self.negative_label, self.positive_label]
        
        self.num_pos_label= np.sum(labels == self.positive_label)
        self.num_neg_label= np.sum(labels == self.negative_label)
        
        self.nbrs= NearestNeighbors(n_neighbors= (self.cutoff or len(X))).fit(self.X)
        
        self.D= float(len(labels))
        
        self.positive_freq= sum(self.labels == self.positive_label)/self.D
        self.negative_freq= 1 - self.positive_freq
        
        self.L_g= self.eq1(self.c_g, self.positive_freq, self.D)
        
    def predict(self, X):
        """
        Predict function following the usual sklearn interface
        Args:
            X (np.ndarray): matrix of unseen vectors for prediction
        Returns:
            np.array: class labels
        """
        
        return np.array([map(lambda x: self.classes_[0] if x[0] > x[1] else self.classes_[1], self.predict_proba(X))])
    
    def predict_proba(self, X):
        """
        Predicts the probabilities following the usual sklearn interface
        Args:
            X (np.ndarray): matrix of unseen vectors for prediction
        Returns:
            np.matrix: matrix of class probabilites
        """

        distances, indices= self.nbrs.kneighbors(X)
        
        if isinstance(X, pd.DataFrame):
            X= X.as_matrix()
        
        print(distances.shape)
        print(indices.shape)
        print(X.shape)
        
        return np.array([self._predict_proba_one_sample(z, d, i) for z, d, i in list(zip(X, distances, indices))])
        
    def _predict_proba_one_sample(self, z, distances, indices):
        """
        Predicts the class probabilities for one sample.
        Args:
            z (np.array): unseen vector to predict
        Returns:
            list: list of class probabilites
        """

        # determining the dynamic query neighborhood
        G= list(zip(distances, indices, self.labels[indices]))
        
        k_pos, r= 0, 0
        while r < len(G)-1:
            if G[r][2] == self.positive_label:
                k_pos= k_pos + 1
            r= r + 1
            if k_pos >= self.n_neighbors and G[r][2] != G[r-1][2]:
                break
        
        # extracting the local confidence interval
        L_r= self.eq2(self.c_r, float(k_pos)/r, float(r))
        
        # checking if density correction needs to be applied
        if L_r[0] > self.L_g[1]:
            if r > k_pos and k_pos != 0:
                lam= (float(k_pos)/float(r - k_pos))/(self.positive_freq/self.negative_freq)
            else:
                lam= 1.0
            p_pos= self.eq4(k_pos, r, self.D, lam)
        else:
            p_pos= self.eq3(k_pos, r, self.D)
        
        # adding probability correction according to the ranking of point patterns
        if self.correction == None:
            pos_prob= p_pos
        elif self.correction == 'r1':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r1(z, indices[:r])))
        elif self.correction == 'r2':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r2(z, indices[:r])))
        elif self.correction == 'r3':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r3(z, indices[:r])))
        elif self.correction == 'pure_r1':
            pos_prob= self.r1(z, indices[:r])
        elif self.correction == 'pure_r2':
            pos_prob= self.r2(z, indices[:r])
        elif self.correction == 'pure_r3':
            pos_prob= self.r3(z, indices[:r])
        elif self.correction == 'r_all':
            r1_tmp= self.r1(z, indices[:r])
            r2_tmp= self.r2(z, indices[:r])
            r3_tmp= self.r3(z, indices[:r])
            r_tmp= (r1_tmp + r2_tmp + r3_tmp)/3.0
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*r_tmp))
        
        return [1 - pos_prob, pos_prob]
