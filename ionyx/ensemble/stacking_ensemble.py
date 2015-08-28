import time
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge

from ..utils.utils import fit_transforms, apply_transforms, score
from ..visualization.visualization import visualize_correlations


def train_stacked_ensemble(X, y, X_test, models, metric, transforms, n_folds):
    """
    Creates a stacked ensemble of many models together.
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
        print('Starting fold {0}...'.format(i + 1))
        X_out_train = X[train_out_index]
        y_out_train = y[train_out_index]
        X_out_eval = X[eval_out_index]
        y_out_eval = y[eval_out_index]

        y_oos = np.zeros((n_records, n_models))

        print('Generating out-of-sample predictions for first-level models...')
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

        print('Fitting second-level model...')
        stacker.fit(y_oos[train_out_index], y_out_train)

        print('Re-fitting first-level models...')
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

        print('Generating predictions and scoring...')
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
    print('Ensemble training completed in {0:3f} s.'.format(t1 - t0))

    for k, model in enumerate(models):
        print('Model ' + str(k) + ' average training score ='), model_train_scores[:, k].sum(axis=0) / n_folds
        print('Model ' + str(k) + ' eval score ='), score(y_true, y_models[:, k], metric)
    print('Ensemble average training score ='), stacker_train_scores.sum() / n_folds
    print('Ensemble eval score ='), score(y_true, y_pred, metric)

    df = pd.DataFrame(y_models, columns=['Model ' + str(i) for i in range(n_models)])
    visualize_correlations(df)

    print('Fitting models on full data set...')
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

    print('Generating test data predictions...')
    for k, model in enumerate(models):
        y_models_test[:, k] = model.predict(X_test).ravel()

    y_pred_test = stacker.predict(y_models_test)

    print('Ensemble complete.')
    return y_models, y_true, y_models_test, y_pred_test
