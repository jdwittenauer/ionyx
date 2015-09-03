import time

from sklearn.cross_validation import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.naive_bayes import *
from sklearn.svm import *

from xgboost import *
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.layers.advanced_activations import *
from keras.models import *
from keras.optimizers import *

from ..utils.utils import fit_transforms, apply_transforms, predict_score


def define_model(model_type, algorithm):
    """
    Defines and returns a model object of the designated type.
    """
    model = None

    if model_type == 'classification':
        if algorithm == 'bayes':
            model = GaussianNB()
        elif algorithm == 'logistic':
            model = LogisticRegression(penalty='l2', C=1.0)
        elif algorithm == 'svm':
            model = SVC(C=1.0, kernel='rbf', shrinking=True, probability=False, cache_size=200)
        elif algorithm == 'sgd':
            model = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, n_iter=1000, shuffle=False, n_jobs=-1)
        elif algorithm == 'forest':
            model = RandomForestClassifier(n_estimators=10, criterion='gini', max_features='auto', max_depth=None,
                                           min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'xt':
            model = ExtraTreesClassifier(n_estimators=10, criterion='gini', max_features='auto', max_depth=None,
                                         min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                               max_leaf_nodes=None)
        elif algorithm == 'xgb':
            model = XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True,
                                  objective='multi:softmax', gamma=0, min_child_weight=1, max_delta_step=0,
                                  subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
        else:
            print('No model defined for ' + algorithm)
            exit()
    else:
        if algorithm == 'ridge':
            model = Ridge(alpha=1.0)
        elif algorithm == 'svm':
            model = SVR(C=1.0, kernel='rbf', shrinking=True, cache_size=200)
        elif algorithm == 'sgd':
            model = SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, n_iter=1000, shuffle=False)
        elif algorithm == 'forest':
            model = RandomForestRegressor(n_estimators=10, criterion='mse', max_features='auto', max_depth=None,
                                          min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'xt':
            model = ExtraTreesRegressor(n_estimators=10, criterion='mse', max_features='auto', max_depth=None,
                                        min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, n_jobs=-1)
        elif algorithm == 'boost':
            model = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0,
                                              min_samples_split=2, min_samples_leaf=1, max_depth=3, max_features=None,
                                              max_leaf_nodes=None)
        elif algorithm == 'xgb':
            # model = XGBRegressor(max_depth=3, learning_rate=0.01, n_estimators=1000, silent=True,
            #                      objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
            #                      subsample=1.0, colsample_bytree=1.0, base_score=0.5, seed=0, missing=None)
            xg = XGBRegressor(max_depth=7, learning_rate=0.005, n_estimators=1800, silent=True,
                              objective='reg:linear', gamma=0, min_child_weight=1, max_delta_step=0,
                              subsample=0.9, colsample_bytree=0.8, base_score=0.5, seed=0, missing=None)
            model = BaggingRegressor(base_estimator=xg, n_estimators=10, max_samples=1.0, max_features=1.0,
                                     bootstrap=True, bootstrap_features=False)
        else:
            print('No model defined for ' + algorithm)
            exit()

    return model


def define_nn_model(input_size, layer_size, output_size, n_hidden_layers, init_method, loss_function,
                             input_activation, hidden_activation, output_activation, use_batch_normalization,
                             input_dropout, hidden_dropout, optimization_method):
    """
    Defines and returns a Keras neural network model.
    """
    model = Sequential()

    # add input layer
    model.add(Dense(input_size, layer_size, init=init_method))

    if input_activation == 'prelu':
        model.add(PReLU((layer_size,)))
    else:
        model.add(Activation(input_activation))

    if use_batch_normalization:
        model.add(BatchNormalization((layer_size,)))

    model.add(Dropout(input_dropout))

    # add hidden layers
    for i in range(n_hidden_layers):
        model.add(Dense(layer_size, layer_size, init=init_method))

        if hidden_activation == 'prelu':
            model.add(PReLU((layer_size,)))
        else:
            model.add(Activation(hidden_activation))

        if use_batch_normalization:
            model.add(BatchNormalization((layer_size,)))

        model.add(Dropout(hidden_dropout))

    # add output layer
    model.add(Dense(layer_size, output_size, init=init_method))
    model.add(Activation(output_activation))

    # configure optimization method
    if optimization_method == 'sgd':
        optimizer = SGD(lr=0.1, momentum=0.9, decay=1e-6, nesterov=True)
    elif optimization_method == 'adagrad':
        optimizer = Adagrad()
    elif optimization_method == 'adadelta':
        optimizer = Adadelta()
    elif optimization_method == 'rmsprop':
        optimizer = RMSprop()
    elif optimization_method == 'adam':
        optimizer = Adam()
    else:
        raise Exception('Optimization method not recognized.')

    model.compile(loss=loss_function, optimizer=optimizer)

    return model


def train_model(X, y, algorithm, model, metric, transforms, early_stopping):
    """
    Trains a new model using the training data.
    """
    if algorithm == 'xgb':
        return train_xgb_model(X, y, model, metric, transforms, early_stopping)
    elif algorithm == 'nn':
        return train_nn_model(X, y, model, metric, transforms, early_stopping)
    else:
        t0 = time.time()
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)
        t1 = time.time()
        print('Model trained in {0:3f} s.'.format(t1 - t0))

        print('Model hyper-parameters:')
        print(model.get_params())

        print('Calculating training score...')
        model_score = predict_score(X, y, model, metric)
        print('Training score ='), model_score

        return model


def train_xgb_model(X, y, model, metric, transforms, early_stopping):
    """
    Trains a new model XGB using the training data.
    """
    t0 = time.time()

    if early_stopping:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)
        model.fit(X_train, y_train, eval_set=[(X_eval, y_eval)], eval_metric='rmse',
                  early_stopping_rounds=100)
        print('Best iteration found: ' + str(model.best_iteration))

        print('Re-fitting at the new stopping point...')
        model.n_estimators = model.best_iteration
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        model.fit(X, y)

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_params())

    print('Calculating training score...')
    model_score = predict_score(X, y, model, metric)
    print('Training score ='), model_score

    return model


def train_nn_model(X, y, model, metric, transforms, early_stopping):
    """
    Trains a new Keras model using the training data.
    """
    t0 = time.time()
    X_train = None
    X_eval = None
    y_train = None
    y_eval = None

    print('Beginning training...')
    if early_stopping:
        X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.1)
        transforms = fit_transforms(X_train, y_train, transforms)
        X_train = apply_transforms(X_train, transforms)
        X_eval = apply_transforms(X_eval, transforms)
        # eval_monitor = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        # history = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=0,
        #                     validation_data=(X_eval, y_eval), shuffle=True, callbacks=[eval_monitor])
        history = model.fit(X_train, y_train, batch_size=128, nb_epoch=100, verbose=0,
                            validation_data=(X_eval, y_eval), shuffle=True)
    else:
        transforms = fit_transforms(X, y, transforms)
        X = apply_transforms(X, transforms)
        history = model.fit(X, y, batch_size=128, nb_epoch=100, verbose=0, shuffle=True, callbacks=[])

    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_config())

    print('Min eval loss ='), min(history.history['val_loss'])
    print('Min eval epoch ='), min(enumerate(history.history['loss']), key=lambda x: x[1])[0] + 1

    if early_stopping:
        print('Calculating training score...')
        train_score = predict_score(X_train, y_train, model, metric)
        print('Training score ='), train_score

        print('Calculating evaluation score...')
        eval_score = predict_score(X_eval, y_eval, model, metric)
        print('Evaluation score ='), eval_score
    else:
        print('Calculating training score...')
        train_score = predict_score(X, y, model, metric)
        print('Training score ='), train_score

    return model, history
