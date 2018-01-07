import numpy as np


class CategoryToNumeric(object):
    """
    Transform class that replaces a categorical value with a representative target value
    for instances that belong to that category.  This technique is useful as a method to
    turn categorical features into numeric values for use in an estimator, and can be
    viewed as an alternative approach to one-hot encoding.  Only suitable for regression
    tasks.

    Parameters
    ----------
    categorical_features : array-like
        A list of integers representing the column indices to apply the transform to.

    metric : {'mean', 'median', 'std'}, optional, default 'mean'
        The method used to calculate the replacement value for a category.

    Attributes
    ----------
    feature_map_ : dict
        Mapping of categorical to target values.
    """
    def __init__(self, categorical_features, metric='mean'):
        self.categorical_features = categorical_features
        self.metric = metric
        self.feature_map_ = {}

    def fit(self, X, y):
        """
        Fit the transform using X as the training data and y as the label.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        for i in self.categorical_features:
            self.feature_map_[i] = {}
            distinct = list(np.unique(X[:, i]))
            for j in distinct:
                if self.metric == 'mean':
                    self.feature_map_[i][j] = y[X[:, i] == j].mean()
                elif self.metric == 'median':
                    self.feature_map_[i][j] = y[X[:, i] == j].median()
                elif self.metric == 'std':
                    self.feature_map_[i][j] = y[X[:, i] == j].std()
                else:
                    raise Exception('Metric not not recognized.')

    def transform(self, X):
        """
        Apply the transform to the data.

        Parameters
        ----------
        X : array-like
            Training input samples.
        """
        X_trans = np.copy(X)
        for i in self.categorical_features:
            distinct = list(np.unique(X_trans[:, i]))
            for j in distinct:
                X_trans[X_trans[:, i] == j, i] = self.feature_map_[i][j]

        return X_trans

    def fit_transform(self, X, y):
        """
        Wrapper method that calls fit and transform sequentially.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        self.fit(X, y)
        return self.transform(X)
