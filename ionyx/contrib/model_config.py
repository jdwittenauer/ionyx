from scipy.stats import uniform, randint
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from prophet_regressor import ProphetRegressor


class ModelConfig(object):
    """
    Provides several helper functions for model definition and parameter tuning.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_model(task, algorithm):
        """
        Defines and returns a model object of the designated type.

        Parameters
        ----------
        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        algorithm : {'bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                     'random_forest', 'extra_trees','gradient_boosting', 'xgboost',
                     'keras', 'prophet'}
            Model algorithm to return an object.
        """
        if task == 'classification':
            if algorithm == 'bayes':
                model = GaussianNB()
            elif algorithm == 'logistic':
                model = LogisticRegression()
            elif algorithm == 'linear_svm':
                model = LinearSVC()
            elif algorithm == 'svm':
                model = SVC()
            elif algorithm == 'random_forest':
                model = RandomForestClassifier()
            elif algorithm == 'extra_trees':
                model = ExtraTreesClassifier()
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingClassifier()
            elif algorithm == 'xgboost':
                model = XGBClassifier()
            elif algorithm == 'keras':
                from keras.wrappers.scikit_learn import KerasClassifier
                from ..contrib.keras_builder import KerasBuilder
                model = KerasClassifier(build_fn=KerasBuilder.build_dense_model)
            else:
                raise Exception('No model defined for {0}'.format(algorithm))
        elif task == 'regression':
            if algorithm == 'ridge':
                model = Ridge()
            elif algorithm == 'elastic_net':
                model = ElasticNet()
            elif algorithm == 'linear_svm':
                model = LinearSVR()
            elif algorithm == 'svm':
                model = SVR()
            elif algorithm == 'random_forest':
                model = RandomForestRegressor()
            elif algorithm == 'extra_trees':
                model = ExtraTreesRegressor()
            elif algorithm == 'gradient_boosting':
                model = GradientBoostingRegressor()
            elif algorithm == 'xgboost':
                model = XGBRegressor()
            elif algorithm == 'keras':
                from keras.wrappers.scikit_learn import KerasRegressor
                from ..contrib.keras_builder import KerasBuilder
                model = KerasRegressor(build_fn=KerasBuilder.build_dense_model)
            elif algorithm == 'prophet':
                model = ProphetRegressor()
            else:
                raise Exception('No model defined for {0}'.format(algorithm))
        else:
            raise Exception('No task defined for {0}'.format(task))

        return model

    @staticmethod
    def grid_search_params(task, algorithm):
        """
        Returns a pre-defined grid of hyper-parameters for the specific model.  Use this
        version for an exhaustive grid search.

        Parameters
        ----------
        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        algorithm : {'bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                     'random_forest', 'extra_trees','gradient_boosting', 'xgboost',
                     'keras', 'prophet'}
            Model algorithm to provide a grid of parameter settings.
        """
        if algorithm == 'logistic':
            param_grid = [
                {
                    'penalty': ['l1', 'l2'],
                    'C': [1, 3, 10, 30, 100, 300, 1000, 3000]
                }
            ]
        elif algorithm == 'linear':
            param_grid = [
                {
                    'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
                }
            ]
        elif algorithm == 'svm':
            param_grid = [
                {
                    'C': [1, 3, 10, 30, 100, 300, 1000, 3000],
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                }
            ]
        elif algorithm == 'random_forest' or algorithm == 'extra_trees':
            param_grid = [
                {
                    'n_estimators': [10, 30, 100, 300],
                    'criterion': ['gini', 'entropy', 'mse', 'mae'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [3, 5, 7, 9, None],
                    'min_samples_split': [2, 10, 30, 100],
                    'min_samples_leaf': [1, 3, 10, 30, 100],
                    'min_impurity_decrease': [0, 3, 10]
                }
            ]
        elif algorithm == 'gradient_boosting':
            param_grid = [
                {
                    'loss': ['deviance', 'exponential', 'ls', 'lad', 'huber', 'quantile'],
                    'learning_rate': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    'n_estimators': [100, 300, 1000],
                    'max_depth': [3, 5, 7, 9, None],
                    'min_samples_split': [2, 10, 30, 100],
                    'min_samples_leaf': [1, 3, 10, 30, 100],
                    'subsample': [1.0, 0.9, 0.8, 0.7],
                    'max_features': ['sqrt', 'log2', None],
                    'min_impurity_decrease': [0, 3, 10]
                }
            ]
        elif algorithm == 'xgboost':
            param_grid = [
                {
                    'max_depth': [3, 5, 7, 9, None],
                    'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    'n_estimators': [100, 300, 1000],
                    'objective': ['reg:linear', 'reg:logistic', 'binary:logistic',
                                  'multi:softmax', 'multi:softprob', 'rank:pairwise'],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'gamma': [0],
                    'min_child_weight': [1, 3, 5, 7, None],
                    'max_delta_step': [0],
                    'subsample': [1.0, 0.9, 0.8, 0.7],
                    'colsample_bytree': [1.0, 0.9, 0.8, 0.7],
                    'colsample_bylevel': [1.0, 0.9, 0.8, 0.7],
                    'reg_alpha': [0],
                    'reg_lambda': [1]
                }
            ]
        elif algorithm == 'keras':
            param_grid = [
                {
                    'layer_size': [32, 64, 128, 256, 512, 1024],
                    'n_hidden_layers': [1, 2, 3, 4, 5, 6],
                    'activation_function': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                            'sigmoid', 'hard_sigmoid', 'linear'],
                    'output_activation': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                          'sigmoid', 'hard_sigmoid', 'linear'],
                    'batch_normalization': [True, False],
                    'dropout': [0, 0.5],
                    'optimizer': ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
                    'loss': ['mean_squared_error',
                             'mean_absolute_error',
                             'mean_absolute_percentage_error',
                             'mean_squared_logarithmic_error',
                             'squared_hinge',
                             'hinge',
                             'categorical_hinge',
                             'logcosh',
                             'categorical_crossentropy',
                             'sparse_categorical_crossentropy',
                             'binary_crossentropy',
                             'kullback_leibler_divergence',
                             'poisson',
                             'cosine_proximity'],
                    'batch_size': [16, 32, 64, 128, 256],
                    'nb_epoch': [10, 30, 100, 300, 1000]
                }
            ]
        elif algorithm == 'prophet':
            param_grid = [
                {
                    'growth': ['linear', 'logistic'],
                    'n_changepoints': [10, 20, 30, 40, 50],
                    'seasonality_prior_scale': [1, 3, 10, 30, 100],
                    'holidays_prior_scale': [1, 3, 10, 30, 100],
                    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                }
            ]
        else:
            raise Exception('No params defined for ' + algorithm)

        return param_grid

    @staticmethod
    def random_search_params(task, algorithm):
        """
        Returns a pre-defined grid of hyper-parameters for the specific model.  Use this
        version for a random search using scipy distributions.

        Parameters
        ----------
        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        algorithm : {'bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                     'random_forest', 'extra_trees','gradient_boosting', 'xgboost',
                     'keras', 'prophet'}
            Model algorithm to provide a grid of parameter settings.
        """
        if algorithm == 'logistic':
            param_grid = [
                {
                    'penalty': ['l1', 'l2'],
                    'C': randint(1, 3000)
                }
            ]
        elif algorithm == 'linear':
            param_grid = [
                {
                    'alpha': uniform(0.0001, 0.9999)
                }
            ]
        elif algorithm == 'svm':
            param_grid = [
                {
                    'C': randint(1, 3000),
                    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                }
            ]
        elif algorithm == 'random_forest' or algorithm == 'extra_trees':
            param_grid = [
                {
                    'n_estimators': randint(10, 300),
                    'criterion': ['gini', 'entropy', 'mse', 'mae'],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 100),
                    'min_samples_leaf': randint(1, 100),
                    'min_impurity_decrease': randint(0, 10)
                }
            ]
        elif algorithm == 'gradient_boosting':
            param_grid = [
                {
                    'loss': ['deviance', 'exponential', 'ls', 'lad', 'huber', 'quantile'],
                    'learning_rate': uniform(0.0001, 0.9999),
                    'n_estimators': randint(100, 1000),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 100),
                    'min_samples_leaf': randint(1, 100),
                    'subsample': uniform(0.7, 0.3),
                    'max_features': ['sqrt', 'log2', None],
                    'min_impurity_decrease': randint(0, 10)
                }
            ]
        elif algorithm == 'xgboost':
            param_grid = [
                {
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.0001, 0.9999),
                    'n_estimators': randint(100, 1000),
                    'objective': ['reg:linear', 'reg:logistic', 'binary:logistic',
                                  'multi:softmax', 'multi:softprob', 'rank:pairwise'],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'min_child_weight': randint(1, 10),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3),
                    'colsample_bylevel': uniform(0.7, 0.3),
                }
            ]
        elif algorithm == 'keras':
            param_grid = [
                {
                    'layer_size': randint(32, 1024),
                    'n_hidden_layers': randint(1, 6),
                    'activation_function': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                            'sigmoid', 'hard_sigmoid', 'linear'],
                    'output_activation': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                          'sigmoid', 'hard_sigmoid', 'linear'],
                    'batch_normalization': [True, False],
                    'dropout': [0, 0.5],
                    'optimizer': ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
                    'loss': ['mean_squared_error',
                             'mean_absolute_error',
                             'mean_absolute_percentage_error',
                             'mean_squared_logarithmic_error',
                             'squared_hinge',
                             'hinge',
                             'categorical_hinge',
                             'logcosh',
                             'categorical_crossentropy',
                             'sparse_categorical_crossentropy',
                             'binary_crossentropy',
                             'kullback_leibler_divergence',
                             'poisson',
                             'cosine_proximity'],
                    'batch_size': randint(16, 256),
                    'nb_epoch': randint(10, 1000)
                }
            ]
        elif algorithm == 'prophet':
            param_grid = [
                {
                    'growth': ['linear', 'logistic'],
                    'n_changepoints': randint(10, 50),
                    'seasonality_prior_scale': uniform(1, 99),
                    'holidays_prior_scale': uniform(1, 99),
                    'changepoint_prior_scale': uniform(0.001, 0.499)
                }
            ]
        else:
            raise Exception('No params defined for ' + algorithm)

        return param_grid
