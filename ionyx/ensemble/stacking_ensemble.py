import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge

from ..utils import print_status_message, fit_transforms, apply_transforms, score
from ..visualization import visualize_correlations


def train_stacked_ensemble(X, y, X_test, models, metric, transforms, n_folds, verbose=False, logger=None):
    """
    Creates an stacked ensemble of many models together.  This function performs several steps.  First, it uses the
    model definitions and other parameters provided as input to do K-fold cross-validation on the data set, training
    and evaluating a stacked model on every iteration.  Second, it fits a stacked model to the full data set using the
    out-of-sample predictions created during cross-validation. Finally, it uses the stacker model to generate
    predictions on the test set.

    Stacking is an ensemble procedure that fits a second-level model using predictions from a set of first-level
    models as the input.  For each first-level model, perform cross-validation to generate out-of-sample (OOS)
    predictions for the whole data set (this is important as we can't use predictions on training data, otherwise
    the stacker will severely overfit).  Once this is complete, Use the OOS predictions as input to the second-level
    model.  Fit the second-level on these predictions and then use it to make predictions on some hold-out data set
    that was not used in cross-validation earlier.

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
    stacker = Ridge()
    n_models = len(models)
    n_records = y.shape[0]

    model_train_scores = np.zeros((n_folds, n_models))
    stacker_train_scores = np.zeros(n_folds)
    y_models = np.zeros((n_records, n_models))
    y_pred = np.zeros(n_records)
    y_true = np.zeros(n_records)

    folds = list(KFold(n_records, n_folds=n_folds, shuffle=True, random_state=1337))
    for i, (train_out_index, eval_out_index) in enumerate(folds):
        print_status_message('Starting fold {0}...'.format(str(i + 1)), verbose, logger)
        X_out_train = X[train_out_index]
        y_out_train = y[train_out_index]
        X_out_eval = X[eval_out_index]
        y_out_eval = y[eval_out_index]

        y_oos = np.zeros((n_records, n_models))

        print_status_message('Generating out-of-sample predictions for first-level models...', verbose, logger)
        for j, (train_index, eval_index) in enumerate(folds):
            if j != i:
                X_train = X[train_index]
                y_train = y[train_index]
                X_eval = X[eval_index]

                transforms = fit_transforms(X_train, y_train, transforms)
                X_train = apply_transforms(X_train, transforms)
                X_eval = apply_transforms(X_eval, transforms)

                for k, model in enumerate(models):
                    if k < 3:
                        model.fit(X_train, y_train)
                    elif k == 3:
                        model.fit(X_train, y_train, batch_size=128, nb_epoch=400, verbose=0, shuffle=True)
                    else:
                        model.fit(X_train, y_train, batch_size=128, nb_epoch=1000, verbose=0, shuffle=True)

                for k, model in enumerate(models):
                    y_oos[eval_index, k] = model.predict(X_eval).ravel()

        print_status_message('Fitting second-level model...', verbose, logger)
        stacker.fit(y_oos[train_out_index], y_out_train)

        print_status_message('Re-fitting first-level models...', verbose, logger)
        transforms = fit_transforms(X_out_train, y_out_train, transforms)
        X_out_train = apply_transforms(X_out_train, transforms)
        X_out_eval = apply_transforms(X_out_eval, transforms)

        for k, model in enumerate(models):
            if k < 3:
                model.fit(X_out_train, y_out_train)
            elif k == 3:
                model.fit(X_out_train, y_out_train, batch_size=128, nb_epoch=400, verbose=0, shuffle=True)
            else:
                model.fit(X_out_train, y_out_train, batch_size=128, nb_epoch=1000, verbose=0, shuffle=True)

        print_status_message('Generating predictions and scoring...', verbose, logger)
        training_predictions = np.zeros((X_out_train.shape[0], n_models))

        for k, model in enumerate(models):
            training_predictions[:, k] = model.predict(X_out_train).ravel()
            model_train_scores[i, k] = score(y_out_train, training_predictions[:, k], metric)

        stacker_train_scores[i] = score(y_out_train, stacker.predict(training_predictions), metric)

        for k, model in enumerate(models):
            y_models[eval_out_index, k] = model.predict(X_out_eval).ravel()

        y_pred[eval_out_index] = stacker.predict(y_models[eval_out_index, :])
        y_true[eval_out_index] = y_out_eval

    t1 = time.time()
    print_status_message('Ensemble training completed in {0:3f} s.'.format(str(t1 - t0)), verbose, logger)

    for k, model in enumerate(models):
        avg_train_score = model_train_scores[:, k].sum(axis=0) / n_folds
        eval_score = score(y_true, y_models[:, k], metric)
        print_status_message('Model {0} average training score = {1}'
                             .format(str(k), str(avg_train_score)), verbose, logger)
        print_status_message('Model {0} eval score = {1}'
                             .format(str(k), str(eval_score)), verbose, logger)
    print_status_message('Ensemble average training score = {0}'
                         .format(str(stacker_train_scores.sum() / n_folds)), verbose, logger)
    print_status_message('Ensemble eval score = {0}'
                         .format(str(score(y_true, y_pred, metric))), verbose, logger)

    df = pd.DataFrame(y_models, columns=['Model ' + str(i) for i in range(n_models)])
    visualize_correlations(df)

    print_status_message('Fitting models on full data set...', verbose, logger)
    n_test_records = X_test.shape[0]
    y_models_test = np.zeros((n_test_records, n_models))

    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)
    X_test = apply_transforms(X_test, transforms)

    for k, model in enumerate(models):
        if k < 3:
            model.fit(X, y)
        elif k == 3:
            model.fit(X, y, batch_size=128, nb_epoch=400, verbose=0, shuffle=True)
        else:
            model.fit(X, y, batch_size=128, nb_epoch=1000, verbose=0, shuffle=True)

    stacker.fit(y_models, y_true)

    print_status_message('Generating test data predictions...', verbose, logger)
    for k, model in enumerate(models):
        y_models_test[:, k] = model.predict(X_test).ravel()

    y_pred_test = stacker.predict(y_models_test)

    print_status_message('Ensemble complete.', verbose, logger)
    return y_models, y_true, y_models_test, y_pred_test
