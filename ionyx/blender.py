import datetime
import time
import pprint as pp
import numpy as np
import pandas as pd
# make_stack_layer is not yet finalized, will be in scikit-learn 0.20
# from sklearn.ensemble import VotingClassifier, make_stack_layer
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from .visualizer import Visualizer
from .contrib.averaging_regressor import AveragingRegressor
from .print_message import PrintMessageMixin


class Blender(PrintMessageMixin):
    """
    Provides a unified API for training, evaluating, and generating predictions from an
    ensemble of diverse, unrelated models.  There are two general strategies to combining
    predictions from independent models - voting/averaging, and stacking.  This class
    aims to support model training, validation, and prediction using both strategies.

    Various ensemble strategies are described in detail here:
    https://mlwave.com/kaggle-ensembling-guide/

    Parameters
    ----------
    models : list of tuples
        A list of tuples of the form (name, model) where "name" is a text descriptor for
        the model.  Each model must be API compatible with scikit-learn estimators.
        Pipelines are also supported.

    scoring_metric : string
        Name of the metric to use to evaluate models.  Text must match a valid metric
        expected by the package being used.

    n_jobs : int, optional, default 1
        Number of parallel processes to use (where functionality is available).

    verbose : boolean, optional, default True
        If true, messages will be written to the console.

    logger : object, optional, default None
        An instantiated log writer with an open file handle.  If provided, messages
        will be written to the log file.

    Attributes
    ----------
    ensemble_ : object
        Fitted ensemble model.  Only available after using one of the "build ensemble"
        functions with retrain set to True.
    """
    def __init__(self, models, scoring_metric, n_jobs=1, verbose=True, logger=None):
        PrintMessageMixin.__init__(self, verbose, logger)
        self.models = models
        self.scoring_metric = scoring_metric
        self.n_jobs = n_jobs
        self.ensemble_ = None

    def build_ensemble(self, X, y, cv, task='classification', retrain=False, **kwargs):
        """
        Construct an ensemble of independent estimators using a voting or averaging
        scheme (depending on if the task is classification or regression) to decide
        on the final prediction.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        cv : object
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.

        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        retrain : boolean, optional, default False
            Re-fit the model at the end using all available data.

        **kwargs : dict, optional
            Arguments to pass to the meta-estimator.  If none are provided then the
            defaults will be used.
        """
        self.print_message('Beginning ensemble training...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()

        if task not in ['classification', 'regression']:
            raise Exception('Invalid value for task.')

        if task == 'classification':
            ensemble = VotingClassifier(self.models, **kwargs)
        else:
            ensemble = AveragingRegressor(self.models, **kwargs)
        results = cross_validate(ensemble, X, y, scoring=self.scoring_metric, cv=cv,
                                 n_jobs=self.n_jobs, verbose=0, return_train_score=True)

        t1 = time.time()
        self.print_message('Ensemble cross-validation complete in {0:3f} s.'.format(t1 - t0))

        train_score = np.mean(results['train_score'])
        test_score = np.mean(results['test_score'])
        self.print_message('Training score = {0}'.format(train_score))
        self.print_message('Cross-validation score = {0}'.format(test_score))

        if retrain:
            self.print_message('Re-training ensemble on full data set...')
            t0 = time.time()
            ensemble.fit(X, y)
            self.ensemble_ = ensemble
            t1 = time.time()
            self.print_message('Ensemble training complete in {0:3f} s.'.format(t1 - t0))

    def build_stacking_ensemble(self, X, y, cv, layer_cv, layer_mask, retrain=False):
        """
        Construct a stacked ensemble of models.  Stacking uses model predictions from
        lower layers as inputs to a model at a higher layer.  This technique can be used
        for both classification and regression.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        cv : object
            Top-level cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.

        layer_cv : object
            A cross-validation strategy used specifically for the stacking layers.  This
            determines how the out-of-sample predictions are generated that get passed to
            the next layer.  Accepts all options considered valid by scikit-learn.

        layer_mask : list
            A list of layer identifiers.  The index corresponds to the index of the model
            tuple in the "models" list.

        retrain : boolean, optional, default False
            Re-fit the model at the end using all available data.
        """
        self.print_message('Beginning stacking ensemble training...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()

        layers = []
        layer_idx = list(set(layer_mask))
        for i in (range(len(layer_idx)) - 1):
            layer_models = [x for idx, x in enumerate(self.models)
                            if layer_mask[idx] == layer_idx[i]]
            layer = make_stack_layer(layer_models, cv=layer_cv, n_jobs=self.n_jobs)
            layers.append(layer)

        layers.append(self.models[-1][1])
        ensemble = make_pipeline(layers)
        results = cross_validate(ensemble, X, y, scoring=self.scoring_metric, cv=cv,
                                 n_jobs=self.n_jobs, verbose=0, return_train_score=True)

        t1 = time.time()
        self.print_message('Ensemble cross-validation complete in {0:3f} s.'.format(t1 - t0))

        train_score = np.mean(results['train_score'])
        test_score = np.mean(results['test_score'])
        self.print_message('Training score = {0}'.format(train_score))
        self.print_message('Cross-validation score = {0}'.format(test_score))

        if retrain:
            self.print_message('Re-training ensemble on full data set...')
            t0 = time.time()
            ensemble.fit(X, y)
            self.ensemble_ = ensemble
            t1 = time.time()
            self.print_message('Ensemble training complete in {0:3f} s.'.format(t1 - t0))

    def model_correlations(self, X, y, cv):
        """
        Visualize how correlated each model's predictions are with one another.
        Correlations are based on out-of-sample predictions using cross-validation to
        compare real-world model performance.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        cv : object
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.
        """
        self.print_message('Beginning model correlation testing...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()

        names, _ = zip(*self.models)
        layer = make_stack_layer(self.models, cv=cv, n_jobs=self.n_jobs)
        predictions = layer.fit_transform(X, y)
        df = pd.DataFrame(predictions, columns=names)
        viz = Visualizer(df)
        viz.visualize_correlations()

        t1 = time.time()
        self.print_message('Correlation testing complete in {0:3f} s.'.format(t1 - t0))
