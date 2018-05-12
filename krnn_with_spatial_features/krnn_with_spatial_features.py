# -*- coding: utf-8 -*-
"""
Imbalanced kNN with locally sensitive decision rule
"""

#%%

#######################
# Importing libraries #
#######################

import collections
import copy
import math
import os
import random
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import arff
from scipy.spatial import distance_matrix
from scipy.stats import ttest_ind
import scipy.stats as st
import scipy.stats.mstats

from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import sklearn

import imblearn.over_sampling as imb_os

#%%

#####################
# Reading test data #
#####################

# TODO(laszlzso): ask for data, update path
base_path= '/home/gykovacs/workspaces/imbalanced-knn/data/'

def encode_column_onehot(column):
    # TODO(laszlzso): check how these encoders work
    lencoder= LabelEncoder().fit(column)
    lencoded= lencoder.transform(column)
    ohencoder= OneHotEncoder(sparse= False).fit(lencoded.reshape(-1, 1))
    ohencoded= ohencoder.transform(lencoded.reshape(-1, 1))

    return ohencoded

def encode_column_median(column, missing_values):
    column= copy.deepcopy(column)
    if np.sum([np.in1d(column, missing_values)]) > 0:
        column[np.in1d(column, missing_values)]= np.median(column[~np.in1d(column, missing_values)].astype(float))
    column= column.astype(float)
    return column.values

def encode_features(data, target= 'target', encoding_threshold= 4, missing_values= ['?', None, 'None']):
    columns= []
    column_names= []

    for c in data.columns:
        sys.stdout.write('Encoding column %s' % c)
        if not c == target:
            # if the column is not the target variable
            n_values= len(np.unique(data[c]))
            sys.stdout.write(' number of values: %d => ' % n_values)

            if n_values == 1:
                # there is no need for encoding
                sys.stdout.write('no encoding\n')
                continue
            elif n_values < encoding_threshold:
                # applying one-hot encoding

                sys.stdout.write('one-hot encoding\n')
                ohencoded= encode_column_onehot(data[c])
                for i in range(ohencoded.shape[1]):
                    columns.append(ohencoded[:,i])
                    column_names.append(str(c) + '_onehot_' + str(i))
            else:
                # applying median encoding
                sys.stdout.write('no encoding, missing values replaced by medians\n')
                columns.append(encode_column_median(data[c], missing_values))
                column_names.append(c)

        if c == target:
            # in the target column the least frequent value is set to 1, the
            # rest is set to 0
            sys.stdout.write(' target variable => least frequent value is 1\n')
            column= copy.deepcopy(data[c])
            val_counts= data[target].value_counts()
            if val_counts.values[0] < val_counts.values[1]:
                mask= (column == val_counts.index[0])
                column[mask]= 1
                column[~(mask)]= 0
            else:
                mask= (column == val_counts.index[0])
                column[mask]= 0
                column[~(mask)]= 1

            columns.append(column.astype(int).values)
            column_names.append(target)

    return pd.DataFrame(np.vstack(columns).T, columns= column_names)

def read_hiva():
    db= pd.read_csv(os.path.join(base_path, 'hiva', 'hiva_train.data'), sep= ' ', header= None)
    del db[db.columns[-1]]
    target= pd.read_csv(os.path.join(base_path, 'hiva', 'hiva_train.labels'), header= None)
    db['target']= target

    return encode_features(db)

def read_hypothyroid():
    db= pd.read_csv(os.path.join(base_path, 'hypothyroid', 'hypothyroid.data.txt'), header= None)
    db.columns= ['target'] + list(db.columns[1:])

    return encode_features(db)

def read_sylva():
    db= pd.read_csv(os.path.join(base_path, 'sylva', 'sylva_train.data'), sep= ' ', header= None)
    del db[db.columns[-1]]
    target= pd.read_csv(os.path.join(base_path, 'sylva', 'sylva_train.labels'), header= None)
    db['target']= target

    return encode_features(db)

def read_pc1():
    data, meta= arff.loadarff(os.path.join(base_path, 'pc1', 'pc1.arff'))
    db= pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)

    db.columns= list(db.columns[:-1]) + ['target']
    return encode_features(db)

def read_cm1():
    data, meta= arff.loadarff(os.path.join(base_path, 'cm1', 'cm1.arff.txt'))
    db= pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)
    db.columns= list(db.columns[:-1]) + ['target']

    return encode_features(db)

def read_kc1():
    data, meta= arff.loadarff(os.path.join(base_path, 'kc1', 'kc1.arff.txt'))
    db= pd.DataFrame(data)
    db.set_value(db['defects'] == b'false', 'defects', False)
    db.set_value(db['defects'] == b'true', 'defects', True)
    db.columns= list(db.columns[:-1]) + ['target']

    return encode_features(db)

def read_spectf():
    db0= pd.read_csv(os.path.join(base_path, 'spect_f', 'SPECTF.train.txt'), header= None)
    db1= pd.read_csv(os.path.join(base_path, 'spect_f', 'SPECTF.test.txt'), header= None)
    db= pd.concat([db0, db1])
    db.columns= ['target'] + list(db.columns[1:])

    return encode_features(db)

def read_hepatitis():
    db= pd.read_csv(os.path.join(base_path, 'hepatitis', 'hepatitis.data.txt'), header= None)
    db.columns= ['target'] + list(db.columns[1:])

    return encode_features(db)

def read_vehicle():
    db0= pd.read_csv(os.path.join(base_path, 'vehicle', 'xaa.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db1= pd.read_csv(os.path.join(base_path, 'vehicle', 'xab.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db2= pd.read_csv(os.path.join(base_path, 'vehicle', 'xac.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db3= pd.read_csv(os.path.join(base_path, 'vehicle', 'xad.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db4= pd.read_csv(os.path.join(base_path, 'vehicle', 'xae.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db5= pd.read_csv(os.path.join(base_path, 'vehicle', 'xaf.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db6= pd.read_csv(os.path.join(base_path, 'vehicle', 'xag.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db7= pd.read_csv(os.path.join(base_path, 'vehicle', 'xah.dat.txt'), sep= ' ', header= None, usecols= range(19))
    db8= pd.read_csv(os.path.join(base_path, 'vehicle', 'xai.dat.txt'), sep= ' ', header= None, usecols= range(19))

    db= pd.concat([db0, db1, db2, db3, db4, db5, db6, db7, db8])

    db.columns= list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 'van', 'target', 'other')

    return encode_features(db)

def read_ada():
    db= pd.read_csv(os.path.join(base_path, 'ada', 'ada_train.data'), sep= ' ', header= None)
    del db[db.columns[-1]]
    target= pd.read_csv(os.path.join(base_path, 'ada', 'ada_train.labels'), header= None)
    db['target']= target

    return encode_features(db)

def read_german():
    db= pd.read_csv(os.path.join(base_path, 'german', 'german.data.txt'), sep= ' ', header= None)
    db.columns= list(db.columns[:-1]) + ['target']

    return encode_features(db, encoding_threshold= 20)

def read_glass():
    db= pd.read_csv(os.path.join(base_path, 'glass', 'glass.data.txt'), header= None)
    db.columns= list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 3, 'target', 0)
    del db[db.columns[0]]

    return encode_features(db)

def read_satimage():
    db0= pd.read_csv(os.path.join(base_path, 'satimage', 'sat.trn.txt'), sep= ' ', header= None)
    db1= pd.read_csv(os.path.join(base_path, 'satimage', 'sat.tst.txt'), sep= ' ', header= None)
    db= pd.concat([db0, db1])
    db.columns= list(db.columns[:-1]) + ['target']
    db.set_value(db['target'] != 4, 'target', 0)

    return encode_features(db)

def statistics():
    for d in dbs:
        name= d
        size= len(dbs[d])
        attr= len(dbs[d].iloc[0]) - 1

        print(d)

        classes= np.unique(dbs[d]['target'])
        pos= np.sum(dbs[d]['target'] == classes[0])
        neg= np.sum(dbs[d]['target'] == classes[1])

        if neg < pos:
            neg, pos= pos, neg
            classes[0], classes[1]= classes[1], classes[0]

        features= copy.deepcopy(dbs[d].columns[dbs[d].columns != 'target'])
        f= dbs[d][features].as_matrix()
        t= dbs[d]['target'].as_matrix()
        prep= preprocessing.StandardScaler().fit(f)
        f= prep.transform(f)
        pos_vectors= f[t == classes[1]]
        neg_vectors= f[t == classes[0]]

        dm= distance_matrix(pos_vectors, pos_vectors)
        dm.sort(axis= 1)
        pd= dm[:,1]
        pos_dists= np.mean(dm[:,1])
        pos_std= np.std(dm[:,1])

        dm= distance_matrix(neg_vectors, neg_vectors)
        dm.sort(axis= 1)
        nd= dm[:,1]
        neg_dists= np.mean(dm[:,1])
        neg_std= np.std(dm[:,1])

        p_value= ttest_ind(pd, nd, equal_var= False)

        print('%15s\t%d\t%d\t%26s\t%d:%d (%.2f)\t%.2f:%.2f\t%.2f:%.2f\t%f' % (name, size, attr, str(classes), pos, neg, float(pos)/size*100, pos_dists, neg_dists, pos_std, neg_std, p_value[1]))

dbs= {}

# TODO(gykovacs): 1600+ dimensions causes math range error,
#                 expression should be simplified
#dbs['hiva']= read_hiva()

# TODO(gykovacs): missing values, categorical
dbs['hypothyroid']= read_hypothyroid()

# Works, very slow with LOOCV
#dbs['sylva']= read_sylva()  # works

dbs['glass']= read_glass()  # works

dbs['pc1']= read_pc1()  # works
dbs['cm1']= read_cm1()  # works
dbs['kc1']= read_kc1()  # works
dbs['spectf']= read_spectf()  # works

# TODO(gykovacs): missing values should be handled
# TODO(laszlzso): does it work in its current state?
dbs['hepatitis']= read_hepatitis()

dbs['vehicle']= read_vehicle()  # works
# TODO(laszlzso): why disabled?
#dbs['ada']= read_ada()  # works

# TODO(gykovacs): categorical attributes need to be handled
#dbs['german']= read_german()

# TODO(laszlzso): why disabled?
#dbs['satimage']= read_satimage()  # works

statistics()

#%%

#################################
# Implementation of classifiers #
#################################

"""
Simple kNN implementation
"""
class kNNSimple:

    # TODO(laszlzso): check how default arguments work.
    def __init__(self, n_neighbors= 5):
        self.n_neighbors= n_neighbors

    def fit(self, X, labels):
        self.X= X
        self.labels= labels

        self.positive_label= 1
        self.negative_label= 0
        self.classes_= [self.negative_label, self.positive_label]

        self.nbrs= NearestNeighbors(n_neighbors= self.n_neighbors).fit(self.X)

    def predict(self, z):
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        distances, indices= self.nbrs.kneighbors(z)
        nearest_labels= self.labels[indices[0]]

        return [[np.sum(nearest_labels == self.negative_label)/float(self.n_neighbors),
                 np.sum(nearest_labels == self.positive_label)/float(self.n_neighbors)]]

#%%

class PNN:
    """
    X. Zhang and Y. Li: A positive-biased Nearest Neighbour Algorithm for Imbalanced
    Classification
    In: Advances in Knowledge Discovery and Data Mining
    pp 293--304
    2013
    """
    def __init__(self, n_neighbors= 5, c= 0.1):
        self.n_neighbors= n_neighbors
        self.c= c

    def fit(self, X, labels):
        self.X= X
        self.labels= labels

        self.nbrs= NearestNeighbors(n_neighbors= self.n_neighbors).fit(self.X)

        self.positive_label= 1
        self.negative_label= 0
        self.classes_= [self.negative_label, self.positive_label]

        self.N= len(labels)

    def eq1(self, c, f, N):
        """
        Equation 1 in the cited paper, estimating the false positive error rate
        """
        z= abs(st.norm.ppf(c))

        num= f + z*z/(2.0*N) + z*math.sqrt(f*(1-f)/N + z*z/(4.0*N*N))
        denom= 1 + z*z/N

        return num/denom

    def predict(self, z):
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        glob_f= 1.0 - sum(self.labels == self.positive_label)/float(len(self.labels))
        delta= self.eq1(self.c, glob_f, self.N)

        distances, indices= self.nbrs.kneighbors(z)
        G= list(zip(distances[0], indices[0], self.labels[indices[0]]))

        p_q, n_q, i= 0, 0, 0
        while i < len(G) and p_q < int(math.ceil(self.n_neighbors/2.0)):
            if G[i][2] == self.positive_label:
                p_q= p_q + 1
            else:
                n_q= n_q + 1
            i= i + 1
        r= n_q + p_q
        e= self.eq1(self.c, float(n_q)/r, r)

        if r > self.n_neighbors and e < delta:
            p_pos= int(math.ceil(self.n_neighbors/2.0))/int(self.n_neighbors)
        else:
            p_pos= float(p_q)/r

        return [[1.0 - p_pos, p_pos]]

#%%

"""
Zhang et al: KRNN: k Rare class Nearest Neighbour classification
"""
class KRNN:
    def __init__(self, n_neighbors= 5, c_g= 0.1, c_r= 0.1):
        self.n_neighbors= n_neighbors
        self.c_g= c_g
        self.c_r= c_r

    def fit(self, X, labels):
        self.X= X
        self.labels= labels

        self.nbrs= NearestNeighbors(n_neighbors= len(X), n_jobs= 6).fit(self.X)

        freqs= collections.Counter(labels).most_common()
        self.positive_label= freqs[-1][0]
        self.negative_label= freqs[-2][0]
        self.classes_= [self.negative_label, self.positive_label]

        self.D= float(len(labels))

        self.positive_freq= sum(self.labels == self.positive_label)/self.D
        self.negative_freq= 1 - self.positive_freq

        self.L_g= self.eq1(self.c_g, self.positive_freq, self.D)

    def eq1(self, c, q, r):
        """
        Equation 1 in the cited paper estimating the global confidence interval
        """

        z= abs(st.norm.ppf(c/2.0))
        return (q - z*math.sqrt(q*(1 - q)/r), q + z*math.sqrt(q*(1 - q)/r))

    def eq2(self, c, q, r):
        """
        Equation 2 in the cited paper estimating the local confidence interval
        """

        z= abs(st.norm.ppf(c/2.0))
        pm= z*math.sqrt(q*(1 - q)/r + z*z/(4.0*r*r))
        denom= 1.0 + z*z/r
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
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        """
        Determining the posterior probabilities
        """
        distances, indices= self.nbrs.kneighbors(z)
        G= list(zip(distances[0], indices[0], self.labels[indices[0]]))

        k_q, i= 0, 0
        while i < len(G)-1 and (k_q < self.n_neighbors or G[i][2] == G[i+1][2]):
            if G[i][2] == self.positive_label:
                k_q= k_q + 1
            i= i + 1
        r= i

        if k_q == r:
            return [[0.0, 1.0]]

        L_r= self.eq2(self.c_r, float(k_q)/r, float(r))

        #if L_r[0] > self.L_g[1]:
        lam= (float(k_q)/float(r - k_q))/(self.positive_freq/self.negative_freq)
        p_pos= self.eq4(k_q, r, self.D, lam)
        #else:
        #p_pos= self.eq3(k_q, r, self.D)

        return [[1.0 - p_pos, p_pos]]

#%%

class ENN:
    """
    Li and Zhang: Improving k nearest neighbor with exemplar generalization for imbalanced classification
    """
    def __init__(self, n_neighbors= 5, c= 0.1):
        self.n_neighbors= n_neighbors
        self.c= c

    def fit(self, X, labels):
        self.X= X
        self.labels= labels
        self.nbrs= NearestNeighbors(n_neighbors= len(X)).fit(self.X)

        freqs= collections.Counter(labels).most_common()
        self.positive_label= freqs[-1][0]
        self.negative_label= freqs[-2][0]
        self.classes_= [self.negative_label, self.positive_label]

        self.N= float(len(labels))
        self.positive_freq= sum(self.labels == self.positive_label)/self.N
        self.negative_freq= 1 - self.positive_freq
        self.determine_positive_pivots()

    def eq1(self, c, f, N):
        """
        Equation 1 in the cited paper estimating the false positive error rate
        """
        z= abs(st.norm.ppf(c/2.0))
        num= f + z*z/(2.0*N) + z*math.sqrt(f*(1 - f)/N + z*z/(4.0*N*N))
        denom= 1.0 + z*z/N
        return num/denom

    def determine_positive_pivots(self):
        self.P= {}
        delta= self.eq1(self.c, self.negative_freq, self.N)
        for i in range(len(self.X)):
            distances, indices= self.nbrs.kneighbors(self.X[i].reshape(1, -1))
            G= list(zip(distances[0][1:], indices[0][1:], self.labels[indices[0][1:]]))

            for k in range(1, len(G)):
                if G[k][2] == self.positive_label:
                    break

            r= np.linalg.norm(self.X[i] - self.X[G[k][1]])
            f= float(k - 1)/float(k + 1)
            p= self.eq1(self.c, f, k)
            if p <= delta:
                self.P[i]= r

    def predict(self, z):
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        distances, indices= self.nbrs.kneighbors(z)
        for i in indices[0]:
            if i in self.P:
                distances[0][i]= distances[0][i] - self.P[i]

        G= sorted(zip(distances[0], indices[0], self.labels[indices[0]]), key= lambda x: x[0])

        nearest_labels= [g[2] for g in G[:self.n_neighbors]]

        neg_num= np.sum(nearest_labels == self.negative_label)
        pos_num= np.sum(nearest_labels == self.positive_label)

        return [[neg_num / float(neg_num + pos_num), pos_num / float(neg_num + pos_num)]]

#%%

class OSWrapper():
    def __init__(self, classifier, sampler):
        self.classifier= classifier
        self.sampler= sampler

    def fit(self, X, labels):
        X_sampled, labels_sampled= self.sampler.fit_sample(X, labels)

        self.classifier.fit(X_sampled, labels_sampled)

    def predict(self, z):
        return self.classifier.predict(z)

    def predict_proba(self, z):
        return self.classifier.predict_proba(z)

#%%

class SMOTEWrapper():
    """
    G. Lemaitre, F. Nogueira and C. K. Aridas: Imbalanced-learn: A Python Toolbox
    to Tackle the Curse of Imbalanced Datasets in Machine Learning
    """
    def __init__(self, classifier):
        self.classifier= classifier

    def fit(self, X, labels):
        """
        Samples the dataset by the SMOTE technique and fits the classifier on
        the oversampled data.
        """
        smote= imb_os.SMOTE()

        X_sampled, labels_sampled= smote.fit_sample(X, labels)

        print(scipy.stats.describe(X_sampled).minmax)
        print(scipy.stats.describe(X_sampled).mean)
        print(scipy.stats.describe(X_sampled).variance)

        if random.randint(1, 100) == 1:
            print(len(X_sampled), len(labels_sampled))
        self.classifier.fit(X_sampled, labels_sampled)

    def predict(self, z):
        return self.classifier.predict(z)

    def predict_proba(self, z):
        return self.classifier.predict_proba(z)

#%%

class MetaCostWrapper():
    """
    P. Domingos: MetaCost: A General Method for Making Classifiers Cost-Sensitive
    """

    def __init__(self, classifier, bags= 10, bagsize= 100):
        """
        Default parameters are the same as that of MetaCost in WEKA
        """
        self.classifiers= [sklearn.base.clone(classifier) for b in range(bags)]
        self.bags= bags
        self.bagsize= bagsize

    def fit(self, X, labels):
        """
        Prepares the classifiers for bagging
        """

        freqs= collections.Counter(labels).most_common()
        self.positive_label= freqs[-1][0]
        self.negative_label= freqs[-2][0]
        self.classes_= [self.negative_label, self.positive_label]

        self.neg_to_pos_ratio= float(np.sum(labels == self.negative_label))/np.sum(labels == self.positive_label)

        for b in range(self.bags):
            X_shuffled, labels_shuffled= sklearn.utils.shuffle(X, labels)
            while np.sum(labels_shuffled[:self.bagsize] == self.positive_label) == 0:
                X_shuffled, labels_binary_shuffled= sklearn.utils.shuffle(X, labels)
            self.classifiers[b].fit(X_shuffled[:self.bagsize], labels_shuffled[:self.bagsize])

        # The cost matrix is set according to the proposal in the KRNN paper
        self.cost_matrix= np.ndarray(shape=(2,2), dtype=float)
        self.cost_matrix[0,0]= 0.0
        self.cost_matrix[1,1]= 0.0
        self.cost_matrix[0,1]= self.neg_to_pos_ratio
        self.cost_matrix[1,0]= 1.0

    def predict(self, z):
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        probs= [self.classifiers[b].predict_proba(z.reshape(1,-1)) for b in range(self.bags)]
        quality_probs= np.mean(probs, axis=0)[0]

        cprobs= np.dot(quality_probs, self.cost_matrix)
        cprobs= cprobs/sum(cprobs)

        return [[cprobs[0], cprobs[1]]]

#%% Levi's solution

from scipy import stats

def L2_norm(x):
    return np.sqrt(x.T.dot(x))

def pair_likelihood_pdf( query, pos_sample, neg_sample):
    # returns a normal pdf
    # the weight of the pdf is closer to zero the more positive the query is

    t=query.reshape(-1,1)
    xi=pos_sample.reshape(-1,1)
    xj=neg_sample.reshape(-1,1)

    yi = xi - t
    yj = xj - t
    xij = xi - xj

    # xij length
    d = L2_norm(xij)

    yi_ = xi.T.dot(xij) / d
    yj_ = xj.T.dot(xij) / d

    # distance from xij
    l = np.sqrt(yi.T.dot(yi) - yi_.T.dot(yi_))

    # scaled down by d
    mu = yi_ / d

    alpha = 1.0
    scale = alpha * np.sqrt(l/d)

    return stats.norm(mu,scale)

xi = np.array([[0], [3]])
xj = np.array([[5], [2]])
t = np.array([[0], [0]])

p=pair_likelihood_pdf( query=t, pos_sample=xi, neg_sample=xj)

def get_point_membership_distribution(query, pos_samples, neg_samples):

    return  [
        pair_likelihood_pdf(query,
                             pos_samples[pos_sample_inx, :].T,
                             neg_samples[neg_sample_inx, :].T)
        for neg_sample_inx in np.arange(neg_samples.shape[0])
        for pos_sample_inx in np.arange(pos_samples.shape[0])
    ]

def get_weight(t, pos_samples, neg_samples):

    if len(neg_samples) > 5:
        neg_samples= neg_samples[:5]

    point_membership_distribution=\
        get_point_membership_distribution(t, pos_samples, neg_samples)

    #return np.mean([max(0, min(1, d)) for d in point_membership_distribution])
    lower_bound = min([pdf.ppf(0.01) for pdf in point_membership_distribution])[0]
    upper_bound = max([pdf.ppf(0.99) for pdf in point_membership_distribution])[0]

    if np.isnan(lower_bound) or np.isnan(upper_bound):
        return 0.5

    def eval_pdfs(x):
        return np.sum([pdf.pdf(x) for pdf in point_membership_distribution])

    xs=np.linspace(lower_bound, upper_bound, num=100)
    ys=np.array([eval_pdfs(x) for x in xs])
    ys_pdf=ys / np.sum(ys)
    ys_cdf=np.cumsum(ys_pdf)

    def get_percentile(xs, ys_cdf, percent):
        mask = ys_cdf < percent
        tmp= np.where(mask)[0]
        if len(tmp) == 0:
            return 0.5
        last_true = tmp[-1]
        return xs[last_true]

    return max(0.0, min(1.0, get_percentile(xs, ys_cdf, 0.5)))

#%%

#######################
# The proposed method #
#######################

import scipy.spatial

class kNNLocalDensityCorrection:
    """
    The proposed method
    """
    def __init__(self,
                 n_neighbors= 1,
                 correction= None,
                 w= 0.000001):

        self.n_neighbors= n_neighbors
        self.correction= correction
        self.c_g= 0.1
        self.c_r= 0.1
        self.w= w

    def eq1(self, c, q, r):
        """
        Equation 1 in the cited paper estimating the global confidence interval
        """

        z= abs(st.norm.ppf(c/2.0))
        return (q - z*math.sqrt(q*(1 - q)/r), q + z*math.sqrt(q*(1 - q)/r))

    def eq2(self, c, q, r):
        """
        Equation 2 in the cited paper estimating the local confidence interval
        """

        z= abs(st.norm.ppf(c/2.0))
        pm= z*math.sqrt(q*(1 - q)/r + z*z/(4.0*r*r))
        denom= 1.0 + z*z/r
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
        neg_samples= []
        pos_samples= []

        for i in N:
            if self.labels[i] == self.negative_label:
                neg_samples.append(self.X[i])
            else:
                pos_samples.append(self.X[i])

        #if len(neg_samples) > 20:
        #    mask= np.random.choice(list(range(len(neg_samples))), 20)
        #    neg_samples= [neg_samples[i] for i in range(len(neg_samples)) if i in mask]
        if len(neg_samples) > 10:
            neg_samples= neg_samples[:10]

        if len(neg_samples) <= 1:
            return 1.0

        dist_m= scipy.spatial.distance_matrix(neg_samples, neg_samples)
        dist_m.sort(axis= 1)
        mean_dist= np.mean(dist_m[:,1])
#        max_dist= np.max(np.max(scipy.spatial.distance_matrix(neg_samples, neg_samples)))
        radius= np.linalg.norm(z - neg_samples[-1])

        dist_m2= scipy.spatial.distance_matrix(z.reshape(1,-1), neg_samples)
        mean_dist2= np.mean(dist_m2)

        if abs(mean_dist2) < 0.01:
            return 0.0

        #if radius == 0:
        #    return 1.0

        #if mean_dist > 2*radius:
        #    print(mean_dist, 2*radius)

        #return 1.0 - mean_dist/(2*radius)
        return 1.0 - mean_dist/mean_dist2

    def r3(self, z, N):
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

    def r1(self, z, N):
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
        mean_neg= 0.0
        mean_pos= 0.0
        max_dist= 0.0
        num_neg= 0
        num_pos= 0
        min_neg= 100.0
        min_pos= 100.0
        min_neg_v= None
        min_pos_v= None

        for i in N:
            d= np.linalg.norm(z - self.X[i])
            if self.labels[i] == self.negative_label:
                mean_neg= mean_neg + d
                num_neg= num_neg + 1
                if d < min_neg:
                    min_neg= d
                    min_neg_v= self.X[i]
            else:
                mean_pos= mean_pos + d
                num_pos= num_pos + 1
                if d < min_pos:
                    min_pos= d
                    min_pos_v= self.X[i]

            if d > max_dist:
                max_dist= d

        if num_neg > 0 and min_pos + min_neg > 0:
            v1= min_neg_v - z
            v2= min_pos_v - z

            inner= np.inner(v1, v2)[0][0]
            norm1= np.inner(v1, v1)[0][0]
            norm2= np.inner(v2, v2)[0][0]

            if norm1 > 0 and norm2 > 0:
                angle= (1.0 + inner/(norm1*norm2))/2.0
            elif norm1 > 0:
                angle= 0.0
            else:
                angle= 1.0

            prob= mean_neg/(mean_neg + mean_pos)*2

            return prob
        else:
            return 1.0

    def fit(self, X, labels):
        self.X= X
        self.labels= labels

        # determining the positive class
        self.positive_label= 1
        self.negative_label= 0
        self.classes_= [self.negative_label, self.positive_label]

        self.num_pos_label= np.sum(labels == self.positive_label)
        self.num_neg_label= np.sum(labels == self.negative_label)

        #if len(self.X) > 3500:
        #    self.nbrs= NearestNeighbors(n_neighbors= len(self.X), n_jobs= 6).fit(self.X)
        #else:
        self.nbrs= NearestNeighbors(n_neighbors= len(self.X)).fit(self.X)
        #self.nbrs= NearestNeighbors(n_neighbors= 50).fit(self.X)

        self.D= float(len(labels))

        self.positive_freq= sum(self.labels == self.positive_label)/self.D
        self.negative_freq= 1 - self.positive_freq

        self.L_g= self.eq1(self.c_g, self.positive_freq, self.D)

    def predict(self, z):
        """
        Predicts the class label of z
        """
        prob= self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        """
        Predicts the posterior probabilities of class labels
        """

        distances, indices= self.nbrs.kneighbors(z)

        G= list(zip(distances[0], indices[0], self.labels[indices[0]]))

        k_pos, r= 0, 0
        while r < len(G)-1:
            if G[r][2] == self.positive_label:
                k_pos= k_pos + 1
            r= r + 1
            if k_pos >= self.n_neighbors and G[r][2] != G[r-1][2]:
                break

        L_r= self.eq2(self.c_r, float(k_pos)/r, float(r))

        if L_r[0] > self.L_g[1]:
            if r > k_pos:
                lam= (float(k_pos)/float(r - k_pos))/(self.positive_freq/self.negative_freq)
            else:
                lam= 1.0
            p_pos= self.eq4(k_pos, r, self.D, lam)
        else:
            p_pos= self.eq3(k_pos, r, self.D)

        if self.correction == None:
            pos_prob= p_pos
        elif self.correction == 'r1':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r1(z, indices[0][:r])))
        elif self.correction == 'r2':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r2(z, indices[0][:r])))
        elif self.correction == 'r3':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r3(z, indices[0][:r])))
        elif self.correction == 'r4':
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*self.r4(z, indices[0][:r])))
        elif self.correction == 'pure_r1':
            pos_prob= self.r1(z, indices[0][:r])
        elif self.correction == 'pure_r2':
            pos_prob= self.r2(z, indices[0][:r])
        elif self.correction == 'pure_r3':
            pos_prob= self.r3(z, indices[0][:r])
        elif self.correction == 'pure_r4':
            pos_prob= self.r4(z, indices[0][:r])
        elif self.correction == 'ens_r1':
            pos_prob= 0.5*(p_pos + self.r1(z, indices[0][:r]))
        elif self.correction == 'ens_r2':
            pos_prob= 0.5*(p_pos + self.r2(z, indices[0][:r]))
        elif self.correction == 'ens_r3':
            pos_prob= 0.5*(p_pos + self.r3(z, indices[0][:r]))
        elif self.correction == 'ens_r4':
            pos_prob= 0.5*(p_pos + self.r4(z, indices[0][:r]))
        elif self.correction == 'r_all':
            r1_tmp= self.r1(z, indices[0][:r])
            r3_tmp= self.r3(z, indices[0][:r])
            r4_tmp= self.r4(z, indices[0][:r])
            r_tmp= (r1_tmp + r3_tmp + r4_tmp)/3.0
            pos_prob= max(0.0, min(1, p_pos - self.w/2 + self.w*r_tmp))

        return [[1 - pos_prob, pos_prob]]

#%%

#################
# Testing logic #
#################

def calculate_measures(prob_scores, pred_labels, true_labels, positive_label):
    tp, tn, fp, fn= 0.0, 0.0, 0.0, 0.0

    for i in range(len(pred_labels)):
        if true_labels[i] == positive_label:
            if pred_labels[i] == true_labels[i]:
                tp= tp + 1
            else:
                fn= fn + 1
        else:
            if pred_labels[i] == true_labels[i]:
                tn= tn + 1
            else:
                fp= fp + 1

    print(tp, tn, fp, fn)

    results= {}
    results['acc']= (tp + tn)/(tp + tn + fp + fn)
    results['sens']= tp / (tp + fn)
    results['spec']= tn / (fp + tn)
    results['ppv']= tp / (tp + fp) if tp + fp > 0 else 0.0
    results['npv']= tn / (tn + fn) if tn + fn > 0 else 0.0
    results['bacc']= (tp/(tp + fn) + tn/(fp + tn))/2.0
    results['f1']= 2*tp/(2*tp + fp + fn)

    # TODO: check the way AUC is computed, using convex hull may be useful
    results['auc']= roc_auc_score(true_labels, np.matrix(prob_scores)[:,1])

    return results

#%%

def loocv(X, labels, classifier):

    positive_label= collections.Counter(labels).most_common()[-1][0]
    negative_label= collections.Counter(labels).most_common()[-2][0]

    labels_binary= np.ndarray(shape=(len(labels)), dtype= int)
    labels_binary[labels == positive_label]= 1
    labels_binary[labels == negative_label]= 0
    labels_binary.astype(int)

    prob_scores= []
    true_labels= []
    pred_labels= []

    for i in range(len(X)):
        print(i)
        training_X= np.vstack((X[:i], X[(i+1):]))
        training_labels= np.hstack((labels_binary[:i], labels_binary[(i+1):]))
        test_X= X[i].reshape(1,-1)
        test_label= labels_binary[i]

        prep= preprocessing.StandardScaler().fit(training_X)

        training_X_preproc= prep.transform(training_X)
        test_X_preproc= prep.transform(test_X)

        classifier.fit(training_X_preproc, training_labels)
        prob= classifier.predict_proba(test_X_preproc)[0]

        if prob[0] > prob[1]:
            pred= 0
        else:
            pred= 1

        prob_scores.append(prob)
        pred_labels.append(pred)
        true_labels.append(test_label)

    return calculate_measures(prob_scores, pred_labels, true_labels, 1)

#%%

def shufflecv(X, labels, classifier, k):

    positive_label= collections.Counter(labels).most_common()[-1][0]
    negative_label= collections.Counter(labels).most_common()[-2][0]

    labels_binary= np.ndarray(shape=(len(labels)), dtype= int)
    labels_binary[labels == positive_label]= 1
    labels_binary[labels == negative_label]= 0
    labels_binary.astype(int)

    prob_scores= []
    true_labels= []
    pred_labels= []

#    random.seed(1)

    #kf= KFold(n_splits= k)
    kf= ShuffleSplit(n_splits= k, test_size= 0.1, random_state= 1)

    j= 0
    for train, test in kf.split(X):
        #print('stage: %d' % j)
        j= j + 1
        training_X= X[train]
        training_labels= labels_binary[train]
        test_X= X[test]
        test_labels= labels_binary[test]

        prep= preprocessing.StandardScaler().fit(training_X)
        training_X_preproc= prep.transform(training_X)

        classifier.fit(training_X_preproc, training_labels)

        for t in range(len(test)):
            test_X_preproc= prep.transform(test_X[t].reshape(1, -1))
            prob= classifier.predict_proba(test_X_preproc)[0]
            if prob[0] > prob[1]:
                pred= 0
            else:
                pred= 1

            prob_scores.append(prob)
            pred_labels.append(pred)
            true_labels.append(test_labels[t])

    return calculate_measures(prob_scores, pred_labels, true_labels, 1)

#%%

def execute_on_all_datasets(classifier, method= 'loocv'):
    results= {}
    for name in dbs:
        print('dataset: %s' % name)

        features= dbs[name].columns[dbs[name].columns != 'target']
        if method == 'loocv':
            results[name]= loocv(dbs[name][features].as_matrix(), dbs[name]['target'].as_matrix(), classifier)
        else:
            results[name]= shufflecv(dbs[name][features].as_matrix(), dbs[name]['target'].as_matrix(), classifier, 20)

        print(results[name])

    return results

#%%

krnn_none= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= None), 'shufflecv')

#%%

krnn_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= 'r1',
                                                         w= 0.1), 'shufflecv')

#%%

krnn_r3= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= 'r3',
                                                         w= 0.1), 'shufflecv')

#%%

krnn_r4= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= 'r4',
                                                         w= 0.1), 'shufflecv')


#%%

krnn_r4= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= 'r_all',
                                                         w= 0.1), 'shufflecv')

#%%

for w in [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001]:
    krnn_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r1', w=w))

krnn_none= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= None), 'shufflecv')


krnn_pure_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'pure_r1', w=0.0001), 'shufflecv')

#%%

print_results([krnn_none, krnn_pure_r1], ['auc', 'acc', 'bacc', 'spec', 'sens'])

#%%

krnn_pure_r2= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'pure_r2', w=0.0001), 'shufflecv')

krnn_none= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                         correction= None), 'shufflecv')

krnn_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r1', w=0.0001), 'shufflecv')

krnn_r2= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r2', w=0.0001), 'shufflecv')

krnn_r1_005= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r1', w=0.05), 'shufflecv')

krnn_r2_005= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r2', w=0.05), 'shufflecv')

krnn_r1_ens= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'ens_r1', w=0.05), 'shufflecv')

krnn_r2_ens= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'ens_r2', w=0.05), 'shufflecv')

#%%

def print_results(datasets, measures):
    rows= []
    for ds_name in datasets[0]:
        rows.append([])
        for ds in datasets:
            for m in measures:
                sys.stdout.write('%.4f & ' % ds[ds_name][m])
                rows[-1].append(ds[ds_name][m])
        sys.stdout.write('\n')

    for i in range(len(rows[0])):
        values= []
        for j in range(len(rows)):
            values.append(rows[j][i])
        sys.stdout.write('%.4f & ' % np.mean(values))
    sys.stdout.write('\n')

#%%

import datetime

results= {}
for w in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
    print(datetime.datetime.now())
    print(w)
    print(w, 'None')
    a= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                                 correction= None, w=w), 'shufflecv')
    print(w, 'r1')
    b= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                                 correction= 'r1', w=w), 'shufflecv')
    print(w, 'r3')
    c= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                                 correction= 'r3', w=w), 'shufflecv')
    print(w, 'r4')
    d= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                                 correction= 'r4', w=w), 'shufflecv')
    print(w, 'r_all')
    e= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                                 correction= 'r_all', w=w), 'shufflecv')

    results[w]= [a, b, c, d, e]

#%%

resultsk= {}
for k in [1, 3, 5, 7]:
    print(k)
    print(k, 'None')
    a= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= k,
                                                                 correction= None), 'shufflecv')
    print(k, 'r1')
    b= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= k,
                                                                 correction= 'r1', w=0.2), 'shufflecv')

    print(k, 'r3')
    c= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= k,
                                                                 correction= 'r3', w=0.2), 'shufflecv')
    print(k, 'r4')
    d= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= k,
                                                                 correction= 'r4', w=0.2), 'shufflecv')

    print(k, 'r_all')
    e= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= k,
                                                                 correction= 'r_all', w=0.2), 'shufflecv')
    resultsk[k]= [a, b, c, d, e]

#%%

def aggregated_results(measure):
    xs= []
    ys1= []
    ys2= []
    ys3= []
    ys4= []
    ys5= []
    for w in results:
        d1= results[w][0]
        d2= results[w][1]
        d3= results[w][2]
        d4= results[w][3]
        d5= results[w][4]
        values1= []
        values2= []
        values3= []
        values4= []
        values5= []
        for ds in d1:
            values1.append(d1[ds][measure])
        for ds in d2:
            values2.append(d2[ds][measure])
        for ds in d3:
            values3.append(d3[ds][measure])
        for ds in d4:
            values4.append(d4[ds][measure])
        for ds in d5:
            values5.append(d5[ds][measure])
        xs.append(w)
        ys1.append(np.mean(values1))
        ys2.append(np.mean(values2))
        ys3.append(np.mean(values3))
        ys4.append(np.mean(values4))
        ys5.append(np.mean(values5))

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.04f & ' % xs[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys1[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys2[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys3[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys4[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        if float(xs[i]) in [0.0001, 0.05, 0.1, 0.4, 0.7, 1.0, 1.3]:
            sys.stdout.write('%.4f & ' % ys5[i])
    sys.stdout.write('\n')

    return xs, ys1, ys2, ys3, ys4, ys5

x, auc_1, auc_2, auc_3, auc_4, auc_5= aggregated_results('auc')
x, f1_1, f1_2, f1_3, f1_4, f1_5= aggregated_results('f1')
x, bacc_1, bacc_2, bacc_3, bacc_4, bacc_5= aggregated_results('bacc')

#%%

def print_results(results):
    for w in results:
        print(w)
        d1= results[w][0]
        d2= results[w][1]
        d3= results[w][2]
        d4= results[w][3]
        d5= results[w][4]
        for ds in d1:
            print("%s %s %.4f" % ('None', ds, d1[ds]['auc']))
        print(np.mean([d1[ds]['auc'] for ds in d1]))
        for ds in d2:
            print("%s %s %.4f" % ('R_1', ds, d2[ds]['auc']))
        print(np.mean([d2[ds]['auc'] for ds in d2]))
        for ds in d3:
            print("%s %s %.4f" % ('R_2', ds, d3[ds]['auc']))
        print(np.mean([d3[ds]['auc'] for ds in d3]))
        for ds in d4:
            print("%s %s %.4f" % ('R_3', ds, d4[ds]['auc']))
        print(np.mean([d4[ds]['auc'] for ds in d4]))
        for ds in d5:
            print("%s %s %.4f" % ('R_all', ds, d5[ds]['auc']))
        print(np.mean([d5[ds]['auc'] for ds in d5]))

#%%

def aggregated_resultsk(measure):
    xs= []
    ys1= []
    ys2= []
    ys3= []
    ys4= []
    ys5= []
    for w in resultsk:
        d1= resultsk[w][0]
        d2= resultsk[w][1]
        d3= resultsk[w][2]
        d4= resultsk[w][3]
        d5= resultsk[w][4]
        values1= []
        values2= []
        values3= []
        values4= []
        values5= []
        for ds in d1:
            values1.append(d1[ds][measure])
        for ds in d2:
            values2.append(d2[ds][measure])
        for ds in d3:
            values3.append(d3[ds][measure])
        for ds in d4:
            values4.append(d4[ds][measure])
        for ds in d5:
            values5.append(d5[ds][measure])
        xs.append(w)
        ys1.append(np.mean(values1))
        ys2.append(np.mean(values2))
        ys3.append(np.mean(values3))
        ys4.append(np.mean(values4))
        ys5.append(np.mean(values5))

    for i in range(len(ys1)):
        sys.stdout.write('%d & ' % xs[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys1[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys2[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys3[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys4[i])
    sys.stdout.write('\n')

    for i in range(len(ys1)):
        sys.stdout.write('%.4f & ' % ys5[i])
    sys.stdout.write('\n')

    return xs, ys1, ys2, ys3, ys4, ys5

#%%

print_results([krnn_none, results[1.0][0], results[0.01][1], krnn_pure_r1, krnn_pure_r2, krnn_r1_ens, krnn_r2_ens], ['auc'])

#%%

x, auc_1, auc_2= aggregated_results('auc')
x, f1_1, f1_2= aggregated_results('f1')
x, bacc_1, bacc_2= aggregated_results('bacc')

plt.figure(figsize=(10,3))
plt.plot(x, auc_1, label='AUC R1', c='r', ls='-')
plt.plot(x, auc_2, label='AUC R2', c='r', ls=':')

plt.plot(x, f1_1, label='F1 R1', c='black', ls='-')
plt.plot(x, f1_2, label='F1 R2', c='black', ls=':')

plt.plot(x, bacc_1, label='BACC R1', c='green', ls='-')
plt.plot(x, bacc_2, label='BACC R2', c='green', ls=':')

plt.xlabel('w')
plt.ylabel('performance')
plt.title('Aggregated performance measures as a function of bandwidth')
plt.legend()

#plt.show()
plt.savefig('../paper/figures/agg.eps', format='eps', dpi=1000, bbox_inches='tight')

#krnn_r1= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
#                                                             correction= 'r1', w=0.1))

#krnn_none= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
#                                                         correction= None))

krnn_r2= execute_on_all_datasets(kNNLocalDensityCorrection(n_neighbors= 1,
                                                             correction= 'r2', w=0.001))

smote_knn= execute_on_all_datasets(SMOTEWrapper(kNNSimple(5)), 'shufflecv')
acos_knn= execute_on_all_datasets(ACOSWrapper(kNNSimple(5), None, 0.001, 10), 'shufflecv')

pnn= execute_on_all_datasets(PNN(1))
krnn1= execute_on_all_datasets(KRNN(17))
enn= execute_on_all_datasets(ENN(1))
smote_knn= execute_on_all_datasets(SMOTEWrapper(kNNSimple(15)))
bmr_knn= execute_on_all_datasets(MetaCostWrapper(KNeighborsClassifier(n_neighbors= 5)))
