import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.externals.joblib import Parallel, delayed
from sklearn.utils import Bunch
from sklearn.utils.metaestimators import _BaseComposition


def _parallel_fit_estimator(estimator, X, y, sample_weight=None):
    if sample_weight:
        estimator.fit(X, y, sample_weight=sample_weight)
    else:
        estimator.fit(X, y)

    return estimator


class AveragingRegressor(_BaseComposition, RegressorMixin):
    """
    Ensemble method that averages the predictions of many independent models together
    to create a final prediction.

    Parameters
    ----------
    estimators : list of (string, estimator) tuples
        List of estimators to fit for the ensemble.

    weights : array-like, shape = (n_estimators), optional, default None
        Sequence of weights (int or float) to weight predictions before averaging.

    n_jobs : int, optional, default 1
        Number of jobs to run in parallel for "fit".  If -1, then the number of jobs
        is set to the number of cores.
    """
    def __init__(self, estimators, weights=None, n_jobs=1):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.estimators_ = None
        self.named_estimators_ = None

    def fit(self, X, y):
        """
        Fit the averaging model.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Training data.
        y : array, shape = (n_samples,)
            Target values

        Returns
        -------
        self : Returns an instance of self.
        """
        name, models = zip(*self.estimators)
        self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_parallel_fit_estimator)(clone(model), X, y)
                for model in models)

        self.named_estimators_ = Bunch(**dict())
        for k, e in zip(self.estimators, self.estimators_):
            self.named_estimators_[k[0]] = e

        return self

    def predict(self, X):
        """
        Predict using the averaging model.

        Parameters
        ----------
        X : array, shape = (n_samples, n_features)
            Instances to generate predictions for.

        Returns
        -------
        C : array, shape = (n_samples,)
            Returns predicted values.
        """
        predictions = np.asarray([model.predict(X) for model in self.estimators_])
        return np.average(predictions, axis=0, weights=self.weights)

    def set_params(self, **params):
        """
        Set the parameters for the AveragingRegressor

        Parameters
        ----------
        params : keyword arguments
            Parameters to set.  Can set parameters for individual models in addition
            to parameters for the ensemble model.
        """
        super(AveragingRegressor, self)._set_params('estimators', **params)
        return self

    def get_params(self, deep=True):
        """
        Get the parameters of the AveragingRegressor.

        Parameters
        ----------
        deep: bool
            Setting it to True gets the various models and model parameters.
        """
        return super(AveragingRegressor, self)._get_params('estimators', deep=deep)
