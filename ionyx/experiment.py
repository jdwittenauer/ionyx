import datetime
import pickle
import time
import pprint as pp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import get_scorer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, learning_curve, train_test_split
from print_message import PrintMessageMixin


class Experiment(PrintMessageMixin):
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
        Name of the metric to use to score models.  Text must match a valid scikit-learn
        metric.

    eval_metric : string
        Separate metric used specifically for evaluation such as hold-out sets during
        training.  Text must match an evaluation metric supported by the package the
        model originates from.

    n_jobs : int, optional, default 1
        Number of parallel processes to use (where functionality is available).

    verbose : boolean, optional, default True
        If true, messages will be written to the console.

    logger : object, optional, default None
        An instantiated log writer with an open file handle.  If provided, messages
        will be written to the log file.

    data : DataFrame, optional, default None
        The data set to be used for the experiment.  Provides the option to specify the
        data at initialization vs. passing in training data and labels with each
        function call. If "data" is specified then "X_columns" and "y_column" must also
        be specified.

    X_cols : list, optional, default None
        List of columns in "data" to use for the training set.

    y_col : string, optional, default None
        Name of the column in "data" to use as a label for supervised learning.

    cv : object, optional, default None
        A cross-validation strategy.  Accepts all options considered valid by
        scikit-learn.

    Attributes
    ----------
    scorer_ : object
        Scikit-learn scoring function for the provided scoring metric.

    best_model_ : object
        The best model found during a parameter search.
    """
    def __init__(self, package, model, scoring_metric, eval_metric=None, n_jobs=1,
                 verbose=True, logger=None, data=None, X_cols=None, y_col=None, cv=None):
        PrintMessageMixin.__init__(self, verbose, logger)
        self.package = package
        self.model = model
        self.scoring_metric = scoring_metric
        self.eval_metric = eval_metric
        self.n_jobs = n_jobs
        self.scorer_ = get_scorer(self.scoring_metric)
        self.best_model_ = None
        self.data = data
        if self.data is not None:
            if X_cols and y_col:
                self.X = data[X_cols].values
                self.y = data[y_col].values
            else:
                raise Exception('X and y columns must be specified if data set is provided.')
        self.cv = cv
        self.print_message('Beginning experiment...')
        self.print_message('Package = {0}'.format(package))
        self.print_message('Scoring Metric = {0}'.format(scoring_metric))
        self.print_message('Evaluation Metric = {0}'.format(eval_metric))
        self.print_message('Parallel Jobs = {0}'.format(n_jobs))
        self.print_message('Model:')
        self.print_message(model, pprint=True)
        self.print_message('Parameters:')
        self.print_message(model.get_params(), pprint=True)

    def train_model(self, X=None, y=None, validate=False, early_stopping=False,
                    early_stopping_rounds=None, plot_eval_history=False, fig_size=16):
        """
        Trains a new model using the provided training data.

        Parameters
        ----------
        X : array-like, optional, default None
            Training input samples.  Must be specified if no data was provided during
            initialization.

        y : array-like, optional, default None
            Target values.  Must be specified if no data was provided during
            initialization.

        validate : boolean, optional, default False
            Evaluate model on a hold-out set during training.

        early_stopping : boolean, optional, default False
            Stop training the model when performance on a validation set begins to drop.
            Eval must be enabled.

        early_stopping_rounds : int, optional, default None
            Number of training iterations to allow before stopping training due to
            performance on a validation set. Eval and early_stopping must be enabled.

        plot_eval_history : boolean, optional, default False
            Plot model performance as a function of training time.  Eval must be enabled.

        fig_size : int, optional, default 16
            Size of the evaluation history plot.
        """
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y

        self.print_message('Beginning model training...')
        self.print_message('X dimensions = {0}'.format(self.X.shape))
        self.print_message('y dimensions = {0}'.format(self.y.shape))
        v = 1 if self.verbose else 0
        t0 = time.time()

        if validate and self.package in ['xgboost', 'keras']:
            X_train, X_eval, y_train, y_eval = train_test_split(self.X, self.y, test_size=0.1)
            training_history = None
            min_eval_loss = None
            min_eval_epoch = None

            if early_stopping:
                if self.package == 'xgboost':
                    self.model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
                                   eval_metric=self.eval_metric,
                                   early_stopping_rounds=early_stopping_rounds,
                                   verbose=self.verbose)
                elif self.package == 'keras':
                    from keras.callbacks import EarlyStopping
                    callbacks = [
                        EarlyStopping(monitor='val_loss', patience=early_stopping_rounds)
                    ]
                    training_history = self.model.fit(X_train, y_train, verbose=v,
                                                      validation_data=(X_eval, y_eval),
                                                      callbacks=callbacks)
            else:
                if self.package == 'xgboost':
                    self.model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)],
                                   eval_metric=self.eval_metric, verbose=self.verbose)
                elif self.package == 'keras':
                    training_history = self.model.fit(X_train, y_train, verbose=v,
                                                      validation_data=(X_eval, y_eval))

            if self.package == 'xgboost':
                training_history = self.model.evals_result()['validation_0'][self.eval_metric]
                min_eval_loss = min(training_history)
                min_eval_epoch = training_history.index(min(training_history)) + 1
            elif self.package == 'keras':
                training_history = training_history.history['val_loss']
                min_eval_loss = min(training_history)
                min_eval_epoch = training_history.index(min(training_history)) + 1

            if plot_eval_history:
                df = pd.DataFrame(training_history, columns=['Eval Loss'])
                df.plot(figsize=(fig_size, fig_size * 3 / 4))

            t1 = time.time()
            self.print_message('Model training complete in {0:3f} s.'.format(t1 - t0))
            self.print_message('Training score = {0}'
                               .format(self.scorer_(self.model, X_train, y_train)))
            self.print_message('Min. evaluation score = {0}'.format(min_eval_loss))
            self.print_message('Min. evaluation epoch = {0}'.format(min_eval_epoch))
        elif validate:
            raise Exception('Package does not support evaluation during training.')
        else:
            if self.package == 'keras':
                self.model.set_params(verbose=v)
            self.model.fit(self.X, self.y)
            t1 = time.time()
            self.print_message('Model training complete in {0:3f} s.'.format(t1 - t0))
            self.print_message('Training score = {0}'
                               .format(self.scorer_(self.model, self.X, self.y)))

    def cross_validate(self, X=None, y=None, cv=None):
        """
        Performs cross-validation to estimate the true performance of the model.

        Parameters
        ----------
        X : array-like, optional, default None
            Training input samples.  Must be specified if no data was provided during
            initialization.

        y : array-like, optional, default None
            Target values.  Must be specified if no data was provided during
            initialization.

        cv : object, optional, default None
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.  Must be specified if no cv was passed in during
            initialization.
        """
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        if cv is not None:
            self.cv = cv

        self.print_message('Beginning cross-validation...')
        self.print_message('X dimensions = {0}'.format(self.X.shape))
        self.print_message('y dimensions = {0}'.format(self.y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(self.cv))
        t0 = time.time()
        if self.package == 'keras':
            self.model.set_params(verbose=0)
        results = cross_validate(self.model, self.X, self.y, scoring=self.scoring_metric,
                                 cv=self.cv, n_jobs=self.n_jobs, verbose=0,
                                 return_train_score=True)
        t1 = time.time()
        self.print_message('Cross-validation complete in {0:3f} s.'.format(t1 - t0))

        train_score = np.mean(results['train_score'])
        test_score = np.mean(results['test_score'])
        self.print_message('Training score = {0}'.format(train_score))
        self.print_message('Cross-validation score = {0}'.format(test_score))

    def learning_curve(self, X=None, y=None, cv=None, fig_size=16):
        """
        Plots a learning curve showing model performance against both training and
        validation data sets as a function of the number of training samples.

        Parameters
        ----------
        X : array-like, optional, default None
            Training input samples.  Must be specified if no data was provided during
            initialization.

        y : array-like, optional, default None
            Target values.  Must be specified if no data was provided during
            initialization.

        cv : object, optional, default None
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.  Must be specified if no cv was passed in during
            initialization.

        fig_size : int, optional, default 16
            Size of the plot.
        """
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        if cv is not None:
            self.cv = cv

        self.print_message('Plotting learning curve...')
        self.print_message('X dimensions = {0}'.format(self.X.shape))
        self.print_message('y dimensions = {0}'.format(self.y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(self.cv))
        t0 = time.time()

        if self.package == 'keras':
            self.model.set_params(verbose=0)

        values = learning_curve(self.model, self.X, self.y, cv=self.cv,
                                scoring=self.scoring_metric, n_jobs=self.n_jobs, verbose=0)
        train_sizes, train_scores, test_scores = values
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 3 / 4))
        ax.set_title('Learning Curve')
        ax.set_xlabel('Training Examples')
        ax.set_ylabel('Score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color='b')
        ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color='r')
        ax.plot(train_sizes, train_scores_mean, 'o-', color='b', label='Training score')
        ax.plot(train_sizes, test_scores_mean, 'o-', color='r', label='Cross-validation score')
        ax.legend(loc='best')
        fig.tight_layout()

        t1 = time.time()
        self.print_message('Plot generation complete in {0:3f} s.'.format(t1 - t0))

    def param_search(self, param_grid, X=None, y=None, cv=None, search_type='grid',
                     n_iter=100, save_results_path=None):
        """
        Conduct a search over some pre-defined set of hyper-parameter configurations
        to find the best-performing set of parameter.

        Parameters
        ----------
        param_grid : list, dict
            Parameter search space.  See scikit-learn documentation for GridSearchCV and
            RandomSearchCV for acceptable formatting.

        X : array-like, optional, default None
            Training input samples.  Must be specified if no data was provided during
            initialization.

        y : array-like, optional, default None
            Target values.  Must be specified if no data was provided during
            initialization.

        cv : object, optional, default None
            A cross-validation strategy.  Accepts all options considered valid by
            scikit-learn.  Must be specified if no cv was passed in during
            initialization.

        search_type : {'grid', 'random'}, optional, default 'grid'
            Specifies use of grid search or random search.  Requirements for param_grid
            are different depending on which method is used.  See scikit-learn
            documentation for GridSearchCV and RandomSearchCV for details.

        n_iter : int, optional, default 100
            Number of search iterations to run.  Only applies to random search.

        save_results_path : string, optional, default None
            Specifies a location to save the full results of the search in format.
            File name should end in .csv.
        """
        if X is not None:
            self.X = X
        if y is not None:
            self.y = y
        if cv is not None:
            self.cv = cv

        self.print_message('Beginning hyper-parameter search...')
        self.print_message('X dimensions = {0}'.format(self.X.shape))
        self.print_message('y dimensions = {0}'.format(self.y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(self.cv))
        t0 = time.time()

        if self.package == 'keras':
            self.model.set_params(verbose=0)

        if search_type == 'grid':
            search = GridSearchCV(self.model, param_grid=param_grid, scoring=self.scoring_metric,
                                  n_jobs=self.n_jobs, cv=self.cv, refit=self.scoring_metric,
                                  verbose=0, return_train_score=True)
        elif search_type == 'random':
            search = RandomizedSearchCV(self.model, param_grid, n_iter=n_iter,
                                        scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                        cv=self.cv, refit=self.scoring_metric, verbose=0,
                                        return_train_score=True)
        else:
            raise Exception('Search type not supported.')

        search.fit(self.X, self.y)

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

        if self.package in ['sklearn', 'xgboost', 'prophet']:
            model_file = open(filename, 'rb')
            self.model = pickle.load(model_file)
            model_file.close()
        elif self.package == 'keras':
            from keras.models import load_model
            self.model.model = load_model(filename)
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
            Location of the file to write.  Scikit-learn, XGBoost, and Prophet use
            pickle to write to disk (use .pkl extension for clarity) while Keras has a
            built-in save function that uses the HDF5 file format, so Keras models must
            have a .h5 extension.
        """
        self.print_message('Saving model...')
        t0 = time.time()

        if self.package in ['sklearn', 'xgboost', 'prophet']:
            model_file = open(filename, 'wb')
            pickle.dump(self.model, model_file)
            model_file.close()
        elif self.package == 'keras':
            if hasattr(self.model, 'model'):
                self.model.model.save(filename)
            else:
                raise Exception('Keras model must be fit before saving.')
        else:
            raise Exception('Package not supported.')

        t1 = time.time()
        self.print_message('Model saved in {0:3f} s.'.format(t1 - t0))
