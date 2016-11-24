import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

from ..utils import print_status_message, fit_transforms, apply_transforms, score
from ..visualization import visualize_correlations


def train_averaged_ensemble(X, y, X_test, models, metric, transforms, n_folds, verbose=False, logger=None):
    """
    Creates an averaged ensemble of many models together.  This function performs several steps.  First, it uses the
    model definitions and other parameters provided as input to do K-fold cross-validation on the data set, training
    each model and averaging their predictions on every iteration.  Second, it fits the models to the full data set.
    Finally, it uses the fitted models to generate ensemble predictions on the test set.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    X_test : array-like
        Test input samples.

    models : array-like
        Model definitions for each of the models in the ensemble.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    transforms : array-like
        List of transforms to apply to the input samples.

    n_folds : int
        Number of cross-validation folds to perform.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    y_models : array-like
        Out-of-sample predictions from each model during cross-validation.

    y_true : array-like
        Actual labels for the sample data (ordering lines up with OOS predictions).

    y_models_test : array-like
        Test predictions from each individual model in the ensemble.

    y_pred_test : array-like
        Ensemble test predictions.
    """
    t0 = time.time()
    n_models = len(models)
    n_records = y.shape[0]

    model_train_scores = np.zeros((n_folds, n_models))
    y_models = np.zeros((n_records, n_models))
    y_pred = np.zeros(n_records)
    y_true = np.zeros(n_records)

    folds = list(KFold(n_records, n_folds=n_folds, shuffle=True, random_state=1337))
    for i, (train_index, eval_index) in enumerate(folds):
        print_status_message('Starting fold {0}...'.format(str(i + 1)), verbose, logger)
        X_train = X[train_index]
        y_train = y[train_index]
        X_eval = X[eval_index]
        y_eval = y[eval_index]

        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        print_status_message('Fitting individual models...', verbose, logger)
        for k, model in enumerate(models):
            model.fit(X_train, y_train)

        print_status_message('Generating predictions and scoring...', verbose, logger)
        for k, model in enumerate(models):
            model_train_scores[i, k] = score(y_train, model.predict(X_train), metric)
            y_models[eval_index, k] = model.predict(X_eval)

        y_pred[eval_index] = y_models[eval_index, :].sum(axis=1) / n_models
        y_true[eval_index] = y_eval

    t1 = time.time()
    print_status_message('Ensemble training completed in {0:3f} s.'.format(t1 - t0), verbose, logger)

    for k, model in enumerate(models):
        avg_train_score = model_train_scores[:, k].sum(axis=0) / n_folds
        eval_score = score(y_true, y_models[:, k], metric)
        print_status_message('Model {0} average training score = {1}'
                             .format(str(k), str(avg_train_score)), verbose, logger)
        print_status_message('Model {0} eval score = {1}'.format(str(k), str(eval_score)), verbose, logger)
    print_status_message('Ensemble eval score = {0}'.format(str(score(y_true, y_pred, metric))), verbose, logger)

    df = pd.DataFrame(y_models, columns=['Model ' + str(i) for i in range(n_models)])
    visualize_correlations(df)

    print_status_message('Fitting models on full data set...', verbose, logger)
    n_test_records = X_test.shape[0]
    y_models_test = np.zeros((n_test_records, n_models))

    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)
    X_test = apply_transforms(X_test, transforms)

    for k, model in enumerate(models):
        model.fit(X, y)

    print_status_message('Generating test data predictions...', verbose, logger)
    for k, model in enumerate(models):
        y_models_test[:, k] = model.predict(X_test)

    y_pred_test = y_models_test.sum(axis=1) / n_models

    print_status_message('Ensemble complete.', verbose, logger)
    return y_pred_test
