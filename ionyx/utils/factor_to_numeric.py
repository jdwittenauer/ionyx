import numpy as np


class FactorToNumeric(object):
    def __init__(self, categorical_features=None, metric='mean'):
        self.categorical_features = categorical_features
        self.metric = metric
        self.feature_map_ = {}

    def fit(self, X, y):
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
        X_trans = np.copy(X)
        for i in self.categorical_features:
            distinct = list(np.unique(X_trans[:, i]))
            for j in distinct:
                X_trans[X_trans[:, i] == j, i] = self.feature_map_[i][j]

        return X_trans

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
