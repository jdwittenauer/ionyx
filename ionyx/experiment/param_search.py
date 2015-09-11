import time
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import *
from xgboost import *

from ..utils import fit_transforms, apply_transforms, predict_score
from ..experiment import define_nn_model


def parameter_search(X, y, algorithm, model, metric, transforms, n_folds):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    if algorithm == 'xgb':
        xbg_parameter_search(X, y, metric)
    elif algorithm == 'nn':
        nn_parameter_search(X, y, metric)
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)

        param_grid = None
        if algorithm == 'logistic':
            param_grid = [{'penalty': ['l1', 'l2'], 'C': [0.1, 0.3, 1.0, 3.0]}]
        elif algorithm == 'ridge':
            param_grid = [{'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}]
        elif algorithm == 'svm':
            param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
        elif algorithm == 'sgd':
            param_grid = [{'loss': ['hinge', 'log', 'modified_huber'], 'penalty': ['l1', 'l2'],
                           'alpha': [0.0001, 0.001, 0.01], 'iter': [100, 1000, 10000]}]
        elif algorithm == 'forest' or algorithm == 'xt':
            param_grid = [{'n_estimators': [10, 30, 100, 300], 'criterion': ['gini', 'entropy', 'mse'],
                           'max_features': ['auto', 'log2', None], 'max_depth': [3, 5, 7, 9, None],
                           'min_samples_split': [2, 10, 30, 100], 'min_samples_leaf': [1, 3, 10, 30, 100]}]
        elif algorithm == 'boost':
            param_grid = [{'learning_rate': [0.1, 0.3, 1.0], 'subsample': [1.0, 0.9, 0.7, 0.5],
                           'n_estimators': [100, 300, 1000], 'max_features': ['auto', 'log2', None],
                           'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [2, 10, 30, 100],
                           'min_samples_leaf': [1, 3, 10, 30, 100]}]

        t0 = time.time()
        grid_estimator = GridSearchCV(model, param_grid, scoring=metric, cv=n_folds, n_jobs=1)
        grid_estimator.fit(X, y)
        t1 = time.time()
        print('Grid search completed in {0:3f} s.'.format(t1 - t0))

        print('Best params ='), grid_estimator.best_params_
        print('Best score ='), grid_estimator.best_score_


def xbg_parameter_search(X, y, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    categories = []
    # categories = [3, 4, 5, 6, 7, 8, 10, 11, 14, 15, 16, 19, 21, 27, 28, 29]
    # categories = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18,
    #               19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    transforms = []

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
    transforms = fit_transforms(X_train, y_train, transforms)
    X_train = apply_transforms(X_train, transforms)
    X_eval = apply_transforms(X_eval, transforms)

    for subsample in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for colsample_bytree in [1.0, 0.9, 0.8, 0.7]:
            for max_depth in [3, 5, 7, 9]:
                for min_child_weight in [1, 3, 5, 7]:
                    t0 = time.time()
                    model = XGBRegressor(max_depth=max_depth, learning_rate=0.005, n_estimators=5000, silent=True,
                                         objective='reg:linear', gamma=0, min_child_weight=min_child_weight,
                                         max_delta_step=0, subsample=subsample, colsample_bytree=colsample_bytree,
                                         base_score=0.5, seed=0, missing=None)

                    print('subsample ='), subsample
                    print('colsample_bytree ='), colsample_bytree
                    print('max_depth ='), max_depth
                    print('min_child_weight ='), min_child_weight

                    print('Model hyper-parameters:')
                    print(model.get_params())

                    print('Fitting model...')
                    model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                              early_stopping_rounds=100, verbose=False)
                    print('Best iteration ='), model.best_iteration

                    train_score = predict_score(X_train, y_train, model, metric)
                    print('Training score ='), train_score

                    eval_score = predict_score(X_eval, y_eval, model, metric)
                    print('Evaluation score ='), eval_score

                    t1 = time.time()
                    print('Model trained in {0:3f} s.'.format(t1 - t0))
                    print('')
                    print('')


def nn_parameter_search(X, y, metric):
    """
    Performs an exhaustive search over the specified model parameters.
    """
    transforms = []

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
    transforms = fit_transforms(X_train, y_train, transforms)
    X_train = apply_transforms(X_train, transforms)
    X_eval = apply_transforms(X_eval, transforms)

    init_methods = ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    optimization_methods = ['adagrad', 'adadelta', 'rmsprop', 'adam']
    layer_sizes = [64, 128, 256, 384, 512]
    hidden_layers = [1, 2, 3, 4]
    batch_sizes = [16, 32, 64, 128]

    for init_method in init_methods:
        for optimization_method in optimization_methods:
            for layer_size in layer_sizes:
                for hidden_layer in hidden_layers:
                    for batch_size in batch_sizes:
                        t0 = time.time()
                        print('Compiling model...')
                        model = define_nn_model(input_size=X_train.shape[1],
                                                layer_size=layer_size,
                                                output_size=1,
                                                n_hidden_layers=hidden_layer,
                                                init_method=init_method,
                                                loss_function='mse',
                                                input_activation='prelu',
                                                hidden_activation='prelu',
                                                output_activation='linear',
                                                use_batch_normalization=True,
                                                input_dropout=0.5,
                                                hidden_dropout=0.5,
                                                optimization_method=optimization_method)

                        print('init_method ='), init_method
                        print('optimization_method ='), optimization_method
                        print('layer_size ='), layer_size
                        print('hidden_layer ='), hidden_layer
                        print('batch_size ='), batch_size

                        print('Model hyper-parameters:')
                        print(model.get_config())

                        print('Fitting model...')
                        # eval_monitor = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
                        # history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, verbose=0,
                        #                     validation_data=(X_eval, y_eval), shuffle=True, callbacks=[eval_monitor])
                        history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1000, verbose=0,
                                            validation_data=(X_eval, y_eval), shuffle=True)
                        print('Min eval loss ='), min(history.history['val_loss'])
                        print('Min eval epoch ='), min(enumerate(history.history['loss']), key=lambda x: x[1])[0] + 1

                        train_score = predict_score(X_train, y_train, model, metric)
                        print('Training score ='), train_score

                        eval_score = predict_score(X_eval, y_eval, model, metric)
                        print('Evaluation score ='), eval_score

                        t1 = time.time()
                        print('Model trained in {0:3f} s.'.format(t1 - t0))
                        print('')
                        print('')
