import time
from sklearn.cross_validation import train_test_split

from ..utils import fit_transforms, apply_transforms, predict_score


def train_model(X, y, model, library, metric, transforms, eval=False, plot_eval_history=False,
                early_stopping=False, early_stopping_rounds=None):
    """
    Trains a new model using the provided training data.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    model : object
        An object in memory that represents a model definition.

    library : {'sklearn', 'xgboost', 'keras'}
        The source library of the model.  Supports more than just scikit-learn models, however
        since model APIs can vary there may be different features/capabilities available depending
        on which library is used.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    eval : boolean, optional, default False
        Evaluate model on a hold-out set during training.

    plot_eval_history : boolean, optional, default False
        Plot model performance as a function of training time.  Eval must be enabled.

    early_stopping : boolean, optional, default False
        Stop training the model when performance on a validation set begins to drop. Eval must be enabled.

    early_stopping_rounds : int, optional, default None
        Number of training iterations to allow before stopping training due to performance on a validation set.
        Eval and early_stopping must be enabled.

    Returns
    ----------
    model : object
        An object in memory that represents a fitted model.

    training_history : array-like
        Model performance on a validation set after each training epoch.  Only available for certain models.
    """
    t0 = time.time()
    X_train = None
    X_eval = None
    y_train = None
    y_eval = None
    training_history = None

    if eval:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)

        if early_stopping:
            if library == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                          early_stopping_rounds=early_stopping_rounds)
                training_history = model.eval_results
                print('Best iteration found ='), model.best_iteration
            else:
                raise Exception('Early stopping not supported.')
        else:
            if library == 'xgboost':
                model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse')
                training_history = model.eval_results
                print('TODO')
            elif library == 'keras':
                model.validation_data = (X_eval, y_eval)
                training_history = model.fit(X_train, y_train)
                print('Min eval loss ='), min(training_history.history['val_loss'])
                print('Min eval epoch ='), min(enumerate(training_history.history['loss']), key=lambda x: x[1])[0] + 1
            else:
                raise Exception('Model evaluation not supported.')
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        if library == 'keras':
            training_history = model.fit(X, y)
        else:
            model.fit(X, y)

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_params())

    if eval:
        print('Calculating training score...')
        train_score = predict_score(X_train, y_train, model, metric)
        print('Training score ='), train_score

        print('Calculating evaluation score...')
        eval_score = predict_score(X_eval, y_eval, model, metric)
        print('Evaluation score ='), eval_score

        if plot_eval_history:
            if library == 'xgboost':
                print('TODO')
            elif library == 'keras':
                print('TODO')
            else:
                raise Exception('Eval history not supported.')
    else:
        print('Calculating training score...')
        train_score = predict_score(X, y, model, metric)
        print('Training score ='), train_score

    return model, training_history
