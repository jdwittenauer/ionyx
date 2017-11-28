import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve

from ..utils import print_status_message, fit_transforms, apply_transforms, score


def cross_validate(X, y, model, metric, transforms, n_folds, verbose=False, logger=None):
    """
    Performs cross-validation to estimate the true performance of the model.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    model : object
        An object in memory that represents a model definition.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    n_folds : int
        Number of cross-validation folds.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    cross_validation_score : float
        An aggregated evaluation of the performance of the model on validation data from each fold.
    """
    t0 = time.time()
    y_train_scores = []
    y_pred = np.array([])
    y_true = np.array([])

    folds = list(KFold(y.shape[0], n_folds=n_folds, shuffle=True, random_state=1337))
    for i, (train_index, eval_index) in enumerate(folds):
        print_status_message('Starting fold {0}...'.format(str(i + 1)), verbose, logger)
        X_train = X[train_index]
        y_train = y[train_index]
        X_eval = X[eval_index]
        y_eval = y[eval_index]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        model.fit(X_train, y_train)

        y_train_scores.append(score(y_train, model.predict(X_train), metric))
        y_pred = np.append(y_pred, model.predict(X_eval))
        y_true = np.append(y_true, y_eval)

    t1 = time.time()
    print_status_message('Cross-validation completed in {0:3f} s.'.format(t1 - t0), verbose, logger)

    avg_train_score = sum(y_train_scores) / len(y_train_scores)
    print_status_message('Average training score = {0}'.format(str(avg_train_score)), verbose, logger)

    xval_score = score(y_true, y_pred, metric)
    print_status_message('Cross-validation score = {0}'.format(str(xval_score)), verbose, logger)

    return xval_score


def sequence_cross_validate(X, y, model, metric, transforms, n_folds, strategy='traditional', window_type='fixed',
                            min_window=0, forecast_range=1, plot=False, verbose=False, logger=None):
    """
    Performs time series cross-validation to estimate the true performance of the model.  Normal
    cross-validation can't be applied to time series data since it can't be randomly shuffled.  This
    function uses a sliding window approach to train on a specific time period and then evaluate on a
    slice of time immediately proceeding the training window.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    model : object
        An object in memory that represents a model definition.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    n_folds : int
        Number of cross-validation folds.

    strategy : {'traditional', 'walk_forward'}, optional, default 'traditional'
        Determines how to construct the cross-validation folds.  Using "traditional" will divide the data set
        up evenly.  Using "walk forward" will train and evaluate iteratively on every record (overrides the
        n_folds parameter).

    window_type : {'fixed', 'cumulative'}, optional, default 'fixed'
        Determines if the training window is a fixed size or expands as the folds progress through the data set.

    min_window : int, optional, default 0
        The minimum fold size allowable.

    forecast_range : int, optional, default 1
        Size of the validation set for each fold.

    plot : boolean, optional, default False
        Plot the forecast performance for each fold.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    cross_validation_score : float
        An aggregated evaluation of the performance of the model on validation data from each fold.
    """
    scores = []
    train_count = len(X)

    if strategy == 'walk_forward':
        n_folds = train_count - min_window - forecast_range
        fold_size = 1
    else:
        fold_size = train_count / n_folds

    t0 = time.time()
    for i in range(n_folds):
        if window_type == 'fixed':
            fold_start = i * fold_size
        else:
            fold_start = 0

        fold_end = (i + 1) * fold_size + min_window
        fold_train_end = fold_end - forecast_range

        X_train, X_eval = X[fold_start:fold_train_end, :], X[fold_train_end:fold_end, :]
        y_train, y_eval = y[fold_start:fold_train_end], y[fold_train_end:fold_end]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_eval)
        scores.append(score(y, y_pred, metric))

        if plot is True:
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.set_title('Estimation Error')
            ax.plot(y_pred - y_eval)
            fig.tight_layout()

    t1 = time.time()
    print_status_message('Cross-validation completed in {0:3f} s.'.format(t1 - t0), verbose, logger)

    xval_score = np.mean(scores)
    print_status_message('Cross-validation score = {0}'.format(str(xval_score)), verbose, logger)

    return xval_score


def plot_learning_curve(X, y, model, metric, transforms, n_folds, verbose=False, logger=None):
    """
    Plots a learning curve showing model performance against both training and validation data sets
    as a function of the number of training samples.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    model : object
        An object in memory that represents a model definition.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    n_folds : int
        Number of cross-validation folds.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.
    """
    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)

    t0 = time.time()
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, scoring=metric, cv=n_folds, n_jobs=1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(16, 10))
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
    print_status_message('Learning curve generated in {0:3f} s.'.format(t1 - t0), verbose, logger)
