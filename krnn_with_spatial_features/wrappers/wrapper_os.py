# %% OS wrapper


class OSWrapper():
    def __init__(self, classifier, sampler):
        self.classifier = classifier
        self.sampler = sampler

    def fit(self, X, labels):
        X_sampled, labels_sampled = self.sampler.fit_sample(X, labels)

        self.classifier.fit(X_sampled, labels_sampled)

    def predict(self, z):
        return self.classifier.predict(z)

    def predict_proba(self, z):
        return self.classifier.predict_proba(z)
