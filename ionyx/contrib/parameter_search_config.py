from scipy.stats import *


class ParameterSearchConfig(object):
    def __init__(self):
        pass

    @staticmethod
    def grid_search_config(algorithm):
        if algorithm == 'logistic':
            param_grid = [{'penalty': ['l1', 'l2'], 'C': [0.1, 0.3, 1.0, 3.0]}]
        elif algorithm == 'linear':
            param_grid = [{'alpha': [0.1, 0.3, 1.0, 3.0, 10.0]}]
        elif algorithm == 'svm':
            param_grid = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
                          {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}]
        elif algorithm == 'random_forest' or algorithm == 'extra_trees':
            param_grid = [{'n_estimators': [10, 30, 100, 300], 'criterion': ['gini', 'entropy', 'mse'],
                           'max_features': ['auto', 'log2', None], 'max_depth': [3, 5, 7, 9, None],
                           'min_samples_split': [2, 10, 30, 100], 'min_samples_leaf': [1, 3, 10, 30, 100]}]
        elif algorithm == 'gradient_boosting':
            param_grid = [{'learning_rate': [0.1, 0.3, 1.0], 'subsample': [1.0, 0.9, 0.7, 0.5],
                           'n_estimators': [100, 300, 1000], 'max_features': ['auto', 'log2', None],
                           'max_depth': [3, 5, 7, 9, None], 'min_samples_split': [2, 10, 30, 100],
                           'min_samples_leaf': [1, 3, 10, 30, 100]}]
        elif algorithm == 'xgboost':
            param_grid = [{'max_depth': [3, 5, 7, 9, None], 'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                           'n_estimators': [100, 300, 1000, 3000, 10000], 'min_child_weight': [1, 3, 5, 7, None],
                           'subsample': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5], 'colsample_bytree': [1.0, 0.9, 0.8, 0.7],
                           'colsample_bylevel': [1.0, 0.9, 0.8, 0.7]}]
        elif algorithm == 'keras':
            param_grid = [{'layer_size': [64, 128, 256, 384, 512, 1024], 'n_hidden_layers': [1, 2, 3, 4, 5, 6],
                           'init_method': ['glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
                           'loss_function': ['mse', 'mae', 'mape', 'msle', 'squared_hinge', 'hinge',
                                             'binary_crossentropy', 'categorical_crossentropy'],
                           'input_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                           'hidden_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                           'output_activation': ['sigmoid', 'tanh', 'prelu', 'linear', 'softmax', 'softplus'],
                           'input_dropout': [0, 0.3, 0.5, 0.7], 'hidden_dropout': [0, 0.3, 0.5, 0.7],
                           'optimization_method': ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam'],
                           'batch_size': [16, 32, 64, 128, 256], 'nb_epoch': [10, 30, 100, 300, 1000]}]
        else:
            raise Exception('No params defined for ' + algorithm)

        return param_grid

    @staticmethod
    def random_search_config(algorithm):
        pass
