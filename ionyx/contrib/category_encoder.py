import numpy as np
from sklearn.preprocessing import LabelEncoder


class CategoryEncoder(object):
    """
    Transform class that encodes text-based categorical variables into integers ranging
    from 0 to distinct_values-1, where distinct_values is the number of unique values in
    each category.

    Parameters
    ----------
    categorical_features : list optional, default None
        A list of integers representing the column indices to apply the transform to.
        If None, the transform will attempt to apply itself to all columns with string
        values.

    Attributes
    ----------
    encoders_ : dict
        Fitted encoders.
    """
    def __init__(self, categorical_features=None):
        self.categorical_features = categorical_features
        self.encoders_ = {}

    def fit(self, X):
        """
        Fit the transform using X as the training data.

        Parameters
        ----------
        X : array-like
            Training input samples.
        """
        if self.categorical_features:
            for i in self.categorical_features:
                encoder = LabelEncoder()
                encoder.fit(X[:, i])
                self.encoders_[i] = encoder
        else:
            for i in range(X.shape[1]):
                if type(X[0, i]) in [str, np.str_, np.string_, np.bytes_, np.unicode_]:
                    encoder = LabelEncoder()
                    encoder.fit(X[:, i])
                    self.encoders_[i] = encoder

    def transform(self, X):
        """
        Apply the transform to the data.

        Parameters
        ----------
        X : array-like
            Training input samples.
        """
        X_trans = np.copy(X)
        for index, encoder in self.encoders_.items():
            X_trans[:, index] = encoder.transform(X[:, index])

        return X_trans

    def fit_transform(self, X):
        """
        Wrapper method that calls fit and transform sequentially.

        Parameters
        ----------
        X : array-like
            Training input samples.
        """
        self.fit(X)
        return self.transform(X)
