from scipy.stats import uniform, randint
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA, KernelPCA
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR


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

        Returns
        ----------
        model : object
            Instantiated model object.
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
                from xgboost import XGBClassifier
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
                from xgboost import XGBRegressor
                model = XGBRegressor()
            elif algorithm == 'keras':
                from keras.wrappers.scikit_learn import KerasRegressor
                from ..contrib.keras_builder import KerasBuilder
                model = KerasRegressor(build_fn=KerasBuilder.build_dense_model)
            elif algorithm == 'prophet':
                from prophet_regressor import ProphetRegressor
                model = ProphetRegressor()
            else:
                raise Exception('No model defined for {0}'.format(algorithm))
        else:
            raise Exception('No task defined for {0}'.format(task))

        return model

    @staticmethod
    def model_search_params(search_type, task, algorithm, name=None):
        """
        Returns a pre-defined grid of hyper-parameters for the specific model.

        Parameters
        ----------
        search_type : {'grid', 'random'}
            Specifies use of grid search or random search.  Requirements for param_grid
            are different depending on which method is used.  See scikit-learn
            documentation for GridSearchCV and RandomSearchCV for details.

        task : {'classification', 'regression'}, optional, default 'classification'
            Specifies if the target is continuous or categorical.

        algorithm : {'bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                     'random_forest', 'extra_trees','gradient_boosting', 'xgboost',
                     'keras', 'prophet'}
            Model algorithm to provide a grid of parameter settings.

        name : string, optional, default None
            An arbitrary identifier for the model.  Used when part of a scikit-learn
            pipeline.  If specified, the param_grid will be modified to be compatible
            with use on a pipeline.

        Returns
        ----------
        param_grid : dict
            Hyper-parameter search options.
        """
        if task not in ['classification', 'regression']:
            raise Exception('No task defined for {0}'.format(task))

        if search_type == 'grid':
            if algorithm == 'bayes':
                param_grid = {}
            elif algorithm == 'logistic':
                param_grid = {
                    'penalty': ['l1', 'l2'],
                    'C': [1, 3, 10, 30, 100, 300, 1000, 3000]
                }
            elif algorithm == 'ridge':
                param_grid = {
                    'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
                }
            elif algorithm == 'elastic_net':
                param_grid = {
                    'alpha': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
            elif algorithm == 'linear_svm':
                if task == 'classification':
                    param_grid = {
                        'penalty': ['l1', 'l2'],
                        'C': [1, 3, 10, 30, 100, 300, 1000, 3000]
                    }
                else:
                    param_grid = {
                        'C': [1, 3, 10, 30, 100, 300, 1000, 3000]
                    }
            elif algorithm == 'svm':
                if task == 'classification':
                    param_grid = {
                        'C': [1, 3, 10, 30, 100, 300, 1000, 3000],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                    }
                else:
                    param_grid = {
                        'C': [1, 3, 10, 30, 100, 300, 1000, 3000],
                        'epsilon': [0.001, 0.01, 0.1, 1],
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                    }
            elif algorithm in ['random_forest', 'extra_trees']:
                param_grid = {
                    'n_estimators': [10, 30, 100, 300],
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': [3, 5, 7, 9, None],
                    'min_samples_split': [2, 10, 30, 100],
                    'min_samples_leaf': [1, 3, 10, 30, 100],
                    'min_impurity_decrease': [0, 3, 10]
                }
            elif algorithm == 'gradient_boosting':
                param_grid = {
                    'learning_rate': [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    'n_estimators': [100, 300, 1000],
                    'max_depth': [3, 5, 7, 9, None],
                    'min_samples_split': [2, 10, 30, 100],
                    'min_samples_leaf': [1, 3, 10, 30, 100],
                    'subsample': [1.0, 0.9, 0.8, 0.7],
                    'max_features': ['sqrt', 'log2', None],
                    'min_impurity_decrease': [0, 3, 10]
                }
            elif algorithm == 'xgboost':
                param_grid = {
                    'max_depth': [3, 5, 7, 9, None],
                    'learning_rate': [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    'n_estimators': [100, 300, 1000],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'min_child_weight': [1, 3, 5, 7, None],
                    'subsample': [1.0, 0.9, 0.8, 0.7],
                    'colsample_bytree': [1.0, 0.9, 0.8, 0.7],
                    'colsample_bylevel': [1.0, 0.9, 0.8, 0.7],
                }
            elif algorithm == 'keras':
                param_grid = {
                    'layer_size': [32, 64, 128, 256, 512, 1024],
                    'n_hidden_layers': [1, 2, 3, 4, 5, 6],
                    'activation_function': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                            'sigmoid', 'hard_sigmoid', 'linear'],
                    'output_activation': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                          'sigmoid', 'hard_sigmoid', 'linear'],
                    'batch_normalization': [True, False],
                    'dropout': [0, 0.5],
                    'optimizer': ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
                    'batch_size': [16, 32, 64, 128, 256],
                    'nb_epoch': [10, 30, 100, 300, 1000]
                }
            elif algorithm == 'prophet':
                param_grid = {
                    'growth': ['linear', 'logistic'],
                    'n_changepoints': [10, 20, 30, 40, 50],
                    'seasonality_prior_scale': [1, 3, 10, 30, 100],
                    'holidays_prior_scale': [1, 3, 10, 30, 100],
                    'changepoint_prior_scale': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
                }
            else:
                raise Exception('No model defined for {0}'.format(algorithm))
        elif search_type == 'random':
            if algorithm == 'bayes':
                param_grid = {}
            elif algorithm == 'logistic':
                param_grid = {
                    'penalty': ['l1', 'l2'],
                    'C': randint(1, 3000)
                }
            elif algorithm == 'ridge':
                param_grid = {
                    'alpha': uniform(0.0001, 0.9999)
                }
            elif algorithm == 'elastic_net':
                param_grid = {
                    'alpha': uniform(0.0001, 0.9999),
                    'l1_ratio': uniform(0.1, 0.8)
                }
            elif algorithm == 'linear_svm':
                if task == 'classification':
                    param_grid = {
                        'penalty': ['l1', 'l2'],
                        'C': randint(1, 3000)
                    }
                else:
                    param_grid = {
                        'C': randint(1, 3000)
                    }
            elif algorithm == 'svm':
                if task == 'classification':
                    param_grid = {
                        'C': randint(1, 3000),
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                    }
                else:
                    param_grid = {
                        'C': randint(1, 3000),
                        'epsilon': uniform(0.001, 0.999),
                        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
                    }
            elif algorithm in ['random_forest', 'extra_trees']:
                param_grid = {
                    'n_estimators': randint(10, 300),
                    'max_features': ['sqrt', 'log2', None],
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 100),
                    'min_samples_leaf': randint(1, 100),
                    'min_impurity_decrease': randint(0, 10)
                }
            elif algorithm == 'gradient_boosting':
                param_grid = {
                    'learning_rate': uniform(0.0001, 0.9999),
                    'n_estimators': randint(100, 1000),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 100),
                    'min_samples_leaf': randint(1, 100),
                    'subsample': uniform(0.7, 0.3),
                    'max_features': ['sqrt', 'log2', None],
                    'min_impurity_decrease': randint(0, 10)
                }
            elif algorithm == 'xgboost':
                param_grid = {
                    'max_depth': randint(3, 10),
                    'learning_rate': uniform(0.0001, 0.9999),
                    'n_estimators': randint(100, 1000),
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'min_child_weight': randint(1, 10),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3),
                    'colsample_bylevel': uniform(0.7, 0.3),
                }
            elif algorithm == 'keras':
                param_grid = {
                    'layer_size': randint(32, 1024),
                    'n_hidden_layers': randint(1, 6),
                    'activation_function': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                            'sigmoid', 'hard_sigmoid', 'linear'],
                    'output_activation': ['elu', 'selu', 'softplus', 'softsign', 'relu', 'tanh',
                                          'sigmoid', 'hard_sigmoid', 'linear'],
                    'batch_normalization': [True, False],
                    'dropout': [0, 0.5],
                    'optimizer': ['sgd', 'rmsprop', 'adagrad', 'adadelta', 'adam', 'adamax', 'nadam'],
                    'batch_size': randint(16, 256),
                    'nb_epoch': randint(10, 1000)
                }
            elif algorithm == 'prophet':
                param_grid = {
                    'growth': ['linear', 'logistic'],
                    'n_changepoints': randint(10, 50),
                    'seasonality_prior_scale': uniform(1, 99),
                    'holidays_prior_scale': uniform(1, 99),
                    'changepoint_prior_scale': uniform(0.001, 0.499)
                }
            else:
                raise Exception('No model defined for {0}'.format(algorithm))
        else:
            raise Exception('Invalid search type.')

        if name:
            param_grid = {name + '__' + k: v for k, v in param_grid.items()}
            param_grid[name] = [ModelConfig.get_model(task, algorithm)]

        return param_grid

    @staticmethod
    def get_transform(algorithm):
        """
        Defines and returns a feature selection transform object of the designated type.

        Parameters
        ----------
        algorithm : {'pca', 'kpca', 'grp', 'fa', 'k_best'}
            Transform algorithm to return an object.

        Returns
        ----------
        transform : object
            Instantiated transform object.
        """
        if algorithm == 'pca':
            transform = PCA()
        elif algorithm == 'kpca':
            transform = KernelPCA()
        elif algorithm == 'grp':
            transform = GaussianRandomProjection()
        elif algorithm == 'fa':
            transform = FeatureAgglomeration()
        elif algorithm == 'k_best':
            transform = SelectKBest(mutual_info_regression)
        else:
            raise Exception('No selection algorithm defined for {0}'.format(algorithm))

        return transform

    @staticmethod
    def transform_search_params(search_type, algorithm, n_features, name=None):
        """
        Returns a pre-defined grid of hyper-parameters for the specific transform.

        Parameters
        ----------
        search_type : {'grid', 'random'}
            Specifies use of grid search or random search.  Requirements for param_grid
            are different depending on which method is used.  See scikit-learn
            documentation for GridSearchCV and RandomSearchCV for details.

        algorithm : {'pca', 'kpca', 'grp', 'fa', 'k_best'}
            Transform algorithm to provide a grid of parameter settings.

        n_features : int
            Maximum number of features in the data set.  Used to calculate feature
            subsets for the parameter search.

        name : string, optional, default None
            An arbitrary identifier for the transform.  Used when part of a
            scikit-learn pipeline.  If specified, the param_grid will be modified
            to be compatible with use on a pipeline.

        Returns
        ----------
        param_grid : dict
            Hyper-parameter search options.
        """
        if search_type == 'grid':
            feature_iterations = [
                int(n_features * 0.2),
                int(n_features * 0.4),
                int(n_features * 0.6),
                int(n_features * 0.8),
                n_features
            ]

            if algorithm in ['pca', 'kpca', 'grp']:
                param_grid = {
                    'n_components': feature_iterations
                }
            elif algorithm == 'fa':
                param_grid = {
                    'n_clusters': feature_iterations
                }
            elif algorithm == 'k_best':
                param_grid = {
                    'k': feature_iterations
                }
            else:
                raise Exception('No selection algorithm defined for {0}'.format(algorithm))
        elif search_type == 'random':
            min_features = int(n_features * 0.2)

            if algorithm in ['pca', 'kpca', 'grp']:
                param_grid = {
                    'n_components': randint(min_features, n_features)
                }
            elif algorithm == 'fa':
                param_grid = {
                    'n_clusters': randint(min_features, n_features)
                }
            elif algorithm == 'k_best':
                param_grid = {
                    'k': randint(min_features, n_features)
                }
            else:
                raise Exception('No selection algorithm defined for {0}'.format(algorithm))
        else:
            raise Exception('Invalid search type.')

        if name:
            param_grid = {name + '__' + k: v for k, v in param_grid.items()}
            param_grid[name] = [ModelConfig.get_transform(algorithm)]

        return param_grid
