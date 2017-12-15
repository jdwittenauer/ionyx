import pandas as pd
from sklearn.ensemble import VotingClassifier, make_stack_layer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from . import Visualizer
from ..contrib import AveragingRegressor


class Blender(object):
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
    """
    def __init__(self, models, scoring_metric, n_jobs=1, verbose=True, logger=None):
        self.models = models
        self.scoring_metric = scoring_metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logger = logger
        self.ensemble_ = None

    def print_message(self, message, pprint=False):
        """
        Optionally print a message to the console and/or log to a file.

        Parameters
        ----------
        message : string
            Generic text message.

        pprint : boolean, optional, default False
            Enables stylistic formatting.
        """
        now = datetime.datetime.now().replace(microsecond=0).isoformat(' ')
        if self.verbose:
            if pprint:
                pp.pprint('(' + now + ') ' + message)
            else:
                print('(' + now + ') ' + message)
        if self.logger:
            self.logger.write('(' + now + ') ' + message + '\n')

    def build_ensemble(self, X, y, cv, task='classification', retrain=False, **kwargs):
        """
        Construct an ensemble of independent estimators using a voting or averaging scheme
        (depending on if the task is classification or regression) to decide on the final
        prediction.

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
