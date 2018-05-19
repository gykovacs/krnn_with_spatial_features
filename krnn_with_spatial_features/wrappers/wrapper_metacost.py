# %% MetaCost wrapper


import collections

import numpy as np

import sklearn


# %%


class MetaCostWrapper():
    """
    P. Domingos: MetaCost: A General Method for Making Classifiers
    Cost-Sensitive
    """

    def __init__(self, classifier, bags=10, bagsize=100):
        """
        Default parameters are the same as that of MetaCost in WEKA
        """
        self.classifiers = [sklearn.base.clone(
            classifier) for b in range(bags)]
        self.bags = bags
        self.bagsize = bagsize

    def fit(self, X, labels):
        """
        Prepares the classifiers for bagging
        """

        freqs = collections.Counter(labels).most_common()
        self.positive_label = freqs[-1][0]
        self.negative_label = freqs[-2][0]
        self.classes_ = [self.negative_label, self.positive_label]

        self.neg_to_pos_ratio = float(
            np.sum(labels == self.negative_label))/np.sum(labels == self.positive_label)

        for b in range(self.bags):
            X_shuffled, labels_shuffled = sklearn.utils.shuffle(X, labels)
            while np.sum(labels_shuffled[:self.bagsize] == self.positive_label) == 0:
                X_shuffled, labels_binary_shuffled = sklearn.utils.shuffle(
                    X, labels)
            self.classifiers[b].fit(
                X_shuffled[:self.bagsize], labels_shuffled[:self.bagsize])

        # The cost matrix is set according to the proposal in the KRNN paper
        self.cost_matrix = np.ndarray(shape=(2, 2), dtype=float)
        self.cost_matrix[0, 0] = 0.0
        self.cost_matrix[1, 1] = 0.0
        self.cost_matrix[0, 1] = self.neg_to_pos_ratio
        self.cost_matrix[1, 0] = 1.0

    def predict(self, z):
        prob = self.predict_proba(z)[0]

        if prob[0] > prob[1]:
            return [self.classes_[0]]
        else:
            return [self.classes_[1]]

    def predict_proba(self, z):
        probs = [self.classifiers[b].predict_proba(
            z.reshape(1, -1)) for b in range(self.bags)]
        quality_probs = np.mean(probs, axis=0)[0]

        cprobs = np.dot(quality_probs, self.cost_matrix)
        cprobs = cprobs/sum(cprobs)

        return [[cprobs[0], cprobs[1]]]
