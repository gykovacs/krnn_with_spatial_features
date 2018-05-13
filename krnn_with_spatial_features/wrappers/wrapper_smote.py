# %% SMOTE wrapper


import random

import scipy.stats as st

import imblearn.over_sampling as imb_os


# %%


class SMOTEWrapper():
    """
    G. Lemaitre, F. Nogueira and C. K. Aridas: Imbalanced-learn: A Python
    Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning
    """

    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, X, labels):
        """
        Samples the dataset by the SMOTE technique and fits the classifier on
        the oversampled data.
        """
        smote = imb_os.SMOTE()

        X_sampled, labels_sampled = smote.fit_sample(X, labels)

        print(st.describe(X_sampled).minmax)
        print(st.describe(X_sampled).mean)
        print(st.describe(X_sampled).variance)

        if random.randint(1, 100) == 1:
            print(len(X_sampled), len(labels_sampled))
        self.classifier.fit(X_sampled, labels_sampled)

    def predict(self, z):
        return self.classifier.predict(z)

    def predict_proba(self, z):
        return self.classifier.predict_proba(z)
