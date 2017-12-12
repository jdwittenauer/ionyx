import datetime
import pickle
import time
import pprint as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, learning_curve, train_test_split


class Experiment(object):
    """
    Provides functionality to create and run machine learning experiments.  Designed to
    serve as a "wrapper" for running an experiment.  This class provides methods for
    training, cross-validation, parameter tuning etc.  The main value proposition is
    in providing a simplified API for common tasks, layering useful reporting and logging
    on top, and reconciling capabilities between several popular libraries.

    Parameters
    ----------
    package : {'sklearn', 'xgboost', 'keras', 'prophet'}
        The source package of the model used in the experiment.  Some capabilities are
        available only using certain packages.

    model : object
        An instantiated supervised learning model.  Must be API compatible with
        scikit-learn estimators.  Pipelines are also supported.

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
    def __init__(self, package, model, scoring_metric, n_jobs=1, verbose=True, logger=None):
        self.package = package
        self.model = model
        self.scoring_metric = scoring_metric
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.logger = logger
        self.best_model_ = None
        self.print_message('Beginning experiment...')
        self.print_message('Package = {0}'.format(package))
        self.print_message('Scoring Metric = {0}'.format(scoring_metric))
        self.print_message('Parallel Jobs = {0}'.format(n_jobs))
        self.print_message('Model:')
        self.print_message(model, pprint=True)
        self.print_message('Parameters:')
        self.print_message(model.get_params(), pprint=True)

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

    def train_model(self, X, y, validate=False, early_stopping=False,
                    early_stopping_rounds=None, plot_eval_history=False):
        pass

    def cross_validate(self, X, y, cv):
        pass

    def learning_curve(self, X, y, cv, fig_size=16):
        pass

    def param_search(self, X, y, param_grid, cv, search_type='grid', n_iter=100, save_results_path=None):
        pass

    def load_model(self, filename):
        pass

    def save_model(self, filename):
        pass
