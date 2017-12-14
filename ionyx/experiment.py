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
        """
        Trains a new model using the provided training data.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        validate : boolean, optional, default False
            Evaluate model on a hold-out set during training.

        early_stopping : boolean, optional, default False
            Stop training the model when performance on a validation set begins to drop.
            Eval must be enabled.

        early_stopping_rounds : int, optional, default None
            Number of training iterations to allow before stopping training due to performance
            on a validation set. Eval and early_stopping must be enabled.

        plot_eval_history : boolean, optional, default False
            Plot model performance as a function of training time.  Eval must be enabled.
        """
        self.print_message('Beginning model training...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        t0 = time.time()

        if validate and self.package in ['xgboost', 'keras']:
            X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)

            if early_stopping:
                if self.package == 'xgboost':
                    # TODO
                    self.model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                                   early_stopping_rounds=early_stopping_rounds)
                    training_history = self.model.eval_results
                elif self.package == 'keras':
                    # TODO
                    raise Exception('Not implemented.')
            else:
                if self.package == 'xgboost':
                    # TODO
                    self.model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse')
                    training_history = self.model.eval_results
                elif self.package == 'keras':
                    # TODO
                    training_history = self.model.fit(X_train, y_train, validation_data=(X_eval, y_eval))
                    min_eval_loss = min(training_history.history['val_loss'])
                    min_eval_epoch = min(enumerate(training_history.history['loss']), key=lambda x: x[1])[0] + 1

            if plot_eval_history:
                if self.package == 'xgboost':
                    # TODO
                    raise Exception('Not implemented.')
                elif self.package == 'keras':
                    # TODO
                    raise Exception('Not implemented.')

            t1 = time.time()
            self.print_message('Model training complete in {0:3f} s.'.format(t1 - t0))
            self.print_message('Training score = {0}'.format(self.model.score(X_train)))
            self.print_message('Evaluation score = {0}'.format(self.model.score(X_eval)))
        elif validate:
            raise Exception('Package does not support evaluation during training.')
        else:
            self.model.fit(X, y)
            t1 = time.time()
            self.print_message('Model training complete in {0:3f} s.'.format(t1 - t0))
            self.print_message('Training score = {0}'.format(self.model.score(X)))

    def cross_validate(self, X, y, cv):
        """
        Performs cross-validation to estimate the true performance of the model.

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
        self.print_message('Beginning cross-validation...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()
        results = cross_validate(self.model, X, y, scoring=self.scoring_metric, cv=cv,
                                 n_jobs=self.n_jobs, verbose=0, return_train_score=True)
        t1 = time.time()
        self.print_message('Cross-validation complete in {0:3f} s.'.format(t1 - t0))

        train_score = np.mean(results['train_score'])
        test_score = np.mean(results['test_score'])
        self.print_message('Training score = {0}'.format(train_score))
        self.print_message('Cross-validation score = {0}'.format(test_score))

    def learning_curve(self, X, y, cv, fig_size=16):
        """
        Plots a learning curve showing model performance against both training and validation
        data sets as a function of the number of training samples.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        cv : object
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.

        fig_size : int, optional, default 16
            Size of the plot.
        """
        self.print_message('Plotting learning curve...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()

        train_sizes, train_scores, test_scores = learning_curve(self.model, X, y, scoring=self.scoring_metric,
                                                                cv=cv, n_jobs=self.n_jobs, verbose=0)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 3 / 4))
        ax.set_title('Learning Curve')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                        alpha=0.1, color='b')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                        alpha=0.1, color='r')
        ax.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
        ax.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
        ax.legend(loc='best')
        fig.tight_layout()

        t1 = time.time()
        self.print_message('Plot generation complete in {0:3f} s.'.format(t1 - t0))

    def param_search(self, X, y, param_grid, cv, search_type='grid', n_iter=100, save_results_path=None):
        """
        Conduct a search over some pre-defined set of hyper-parameter configurations
        to find the best-performing set of parameter.

        Parameters
        ----------
        X : array-like
            Training input samples.

        y : array-like
            Target values.

        param_grid : list, dict
            Parameter search space.  See scikit-learn documentation for GridSearchCV and
            RandomSearchCV for acceptable formatting.

        cv : object
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.

        search_type : {'grid', 'random'}, optional, default 'grid'
            Specifies use of grid search or random search.  Requirements for param_grid are
            different depending on which method is used.  See scikit-learn documentation for
            GridSearchCV and RandomSearchCV for details.

        n_iter : int, optional, default 100
            Number of search iterations to run.  Only applies to random search.

        save_results_path : string, optional, default None
            Specifies a location to save the full results of the search in format.
            File name should end in .csv.
        """
        self.print_message('Beginning hyper-parameter search...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        t0 = time.time()

        if search_type == 'grid':
            search = GridSearchCV(self.model, param_grid=param_grid, scoring=self.scoring_metric,
                                  n_jobs=self.n_jobs, cv=cv, refit=self.scoring_metric,
                                  verbose=0, return_train_score=True)
        elif search_type == 'random':
            search = RandomizedSearchCV(self.model, param_grid, n_iter=n_iter, scoring=self.scoring_metric,
                                        n_jobs=self.n_jobs, cv=cv, refit=self.scoring_metric,
                                        verbose=0, return_train_score=True)
        else:
            raise Exception('Search type not supported.')

        search.fit(X, y)

        t1 = time.time()
        self.print_message('Hyper-parameter search complete in {0:3f} s.'.format(t1 - t0))
        self.print_message('Best score found = {0}'.format(search.best_score_))
        self.print_message('Best parameters found = {0}'.format(search.best_params_))
        self.best_model_ = search.best_estimator_

        if save_results_path:
            results = pd.DataFrame(search.cv_results_)
            results = results.sort_values(by='mean_test_score', ascending=False)
            results.to_csv(save_results_path, index=False)

    def load_model(self, filename):
        """
        Load a previously trained model from disk.

        Parameters
        ----------
        filename : string
            Location of the file to read.
        """
        self.print_message('Loading model...')
        t0 = time.time()

        if self.package == 'sklearn':
            model_file = open(filename, 'rb')
            self.model = pickle.load(model_file)
            model_file.close()
        elif self.package == 'xgboost':
            # TODO
            raise Exception('Not implemented.')
        elif self.package == 'keras':
            # TODO
            raise Exception('Not implemented.')
        elif self.package == 'prophet':
            # TODO
            raise Exception('Not implemented.')
        else:
            raise Exception('Package not supported.')

        t1 = time.time()
        self.print_message('Model loaded in {0:3f} s.'.format(t1 - t0))

    def save_model(self, filename):
        """
        Persist a trained model to disk.

        Parameters
        ----------
        filename : string
            Location of the file to write.
        """
        self.print_message('Saving model...')
        t0 = time.time()

        if self.package == 'sklearn':
            model_file = open(filename, 'wb')
            pickle.dump(self.model, model_file)
            model_file.close()
        elif self.package == 'xgboost':
            # TODO
            raise Exception('Not implemented.')
        elif self.package == 'keras':
            # TODO
            raise Exception('Not implemented.')
        elif self.package == 'prophet':
            # TODO
            raise Exception('Not implemented.')
        else:
            raise Exception('Package not supported.')

        t1 = time.time()
        self.print_message('Model saved in {0:3f} s.'.format(t1 - t0))
