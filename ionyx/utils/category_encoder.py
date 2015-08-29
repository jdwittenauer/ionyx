import numpy as np
from sklearn.preprocessing import LabelEncoder


class CategoryEncoder(object):
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features
        self.encoders_ = {}

    def fit(self, X):
        if self.categorical_features is not None:
            for i in self.categorical_features:
                encoder = LabelEncoder()
                encoder.fit(X[:, i])
                self.encoders_[i] = encoder
        else:
            for i in range(X.shape[1]):
                if type(X[0, i]) is str:
                    encoder = LabelEncoder()
                    encoder.fit(X[:, i])
                    self.encoders_[i] = encoder

    def transform(self, X):
        X_trans = np.copy(X)
        for index, encoder in self.encoders_.iteritems():
            X_trans[:, index] = encoder.transform(X[:, index])

        return X_trans

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __repr__(self):
        return '%s' % self.__class__.__name__
