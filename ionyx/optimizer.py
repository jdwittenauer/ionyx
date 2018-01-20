import six
import copy
import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .contrib.model_config import ModelConfig
from .blender import Blender
from .print_message import PrintMessageMixin


def _combine_dict(x, y):
    x.update(y)
    return x


class Optimizer(PrintMessageMixin):
    """
    Provides functionality to automatically find a good model for a given algorithm
    class and data set.  Conducts a parameter search on the model's hyper-parameters
    as well as transforms on the data's feature space.  The result will be a
    scikit-learn pipeline with the best fit transforms and model found by the process.
    Also capable of automatically building an optimized ensemble of models if several
    algorithms are provided as input.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    cv : object
        A cross-validation strategy.  Accepts all options considered valid by
        scikit-learn.

    algorithm : {'bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                 'random_forest', 'extra_trees','gradient_boosting', 'xgboost',
                 'keras', 'prophet'} or list
        Model type to optimize.  Source package will be selected automatically based on
        the algorithm provided.  If a list of algorithms is provided, then an ensemble
        of all of the models together will be created.

    scoring_metric : string
        Name of the metric to use to score models.  Text must match a valid scikit-learn
        metric.

    task : {'classification', 'regression'}, optional, default 'classification'
        Specifies if the target is continuous or categorical.

    search_type : {'grid', 'random'}, optional, default 'grid'
        Specifies use of grid search or random search.  Requirements for param_grid are
        different depending on which method is used.  See scikit-learn documentation for
        GridSearchCV and RandomSearchCV for details.

    n_iter : int, optional, default 100
        Number of search iterations to run.  Only applies to random search.

    n_jobs : int, optional, default 1
        Number of parallel processes to use (where functionality is available).

    verbose : boolean, optional, default True
        If true, messages will be written to the console.

    logger : object, optional, default None
        An instantiated log writer with an open file handle.  If provided, messages
        will be written to the log file.

    Attributes
    ----------
    param_grids_ : dict
        The hyper-parameter grids constructed by the algorithms to conduct a search.
        Returns a dictionary where the keys are the algorithm names and the values
        are the parameter grids.
    best_models_ : dict
        The best fitted pipelines for each algorithm passed to the optimizer.  Returns
        a dictionary where the keys are the algorithm names and the values are the model
        pipelines.
    """
    def __init__(self, X, y, cv, algorithm, scoring_metric, task='classification',
                 search_type='grid', n_iter=100, n_jobs=1, verbose=True, logger=None):
        PrintMessageMixin.__init__(self, verbose, logger)
        self.X = X
        self.y = y
        self.cv = cv
        self.algorithm = algorithm
        self.scoring_metric = scoring_metric
        self.task = task
        self.search_type = search_type
        self.n_iter = n_iter
        self.n_jobs = n_jobs
        self.param_grids_ = {}
        self.best_models_ = {}
        self.print_message('Configuring optimizer...')
        self.print_message('X dimensions = {0}'.format(X.shape))
        self.print_message('y dimensions = {0}'.format(y.shape))
        self.print_message('Cross-validation strategy = {0}'.format(cv))
        self.print_message('Algorithm = {0}'.format(algorithm))
        self.print_message('Scoring Metric = {0}'.format(scoring_metric))
        self.print_message('Task = {0}'.format(task))
        self.print_message('Search Type = {0}'.format(search_type))
        self.print_message('Iterations = {0}'.format(n_iter))
        self.print_message('Parallel Jobs = {0}'.format(n_jobs))

    def _run_single(self, algorithm):
        """
        Runs optimization for a single model.  Should not be called directly (use "run"
        instead).

        Returns
        ----------
        pipeline : object
            The best fit transforms and model discovered during the optimization process.
        """
        if algorithm not in ['bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                             'random_forest', 'extra_trees', 'gradient_boosting', 'xgboost',
                             'keras', 'prophet']:
            raise Exception('Invalid value for algorithm.')

        self.print_message('Initiating process for algorithm = {0}...'.format(algorithm))

        scaler = StandardScaler()
        transform = ModelConfig.get_transform('pca')
        model = ModelConfig.get_model(self.task, algorithm)
        pipe = Pipeline([('scaler', scaler), ('transform', transform), ('model', model)])
        n_features = self.X.shape[1]

        if self.algorithm == 'keras':
            model.set_params(verbose=0)

        self.print_message('Building parameter grid...')
        param_grid = [
            ModelConfig.transform_search_params(self.search_type, 'pca', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'kpca', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'grp', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'fa', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'k_best', n_features, 'transform')
        ]
        model_grid = ModelConfig.model_search_params(self.search_type, self.task,
                                                     algorithm, 'model')
        param_grid = [_combine_dict(x, model_grid) for x in param_grid]

        self.print_message('Initiating search...')
        if self.search_type == 'grid':
            search = GridSearchCV(pipe, param_grid=param_grid, scoring=self.scoring_metric,
                                  n_jobs=self.n_jobs, cv=self.cv, refit=self.scoring_metric,
                                  verbose=0, return_train_score=True)
            search.fit(self.X, self.y)
        elif self.search_type == 'random':
            n_sub_iter = self.n_iter // 5
            search_pca = RandomizedSearchCV(pipe, param_grid[0], n_iter=n_sub_iter,
                                            scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                            cv=self.cv, refit=self.scoring_metric, verbose=0,
                                            return_train_score=True)
            search_pca.fit(self.X, self.y)
            search_kpca = RandomizedSearchCV(pipe, param_grid[1], n_iter=n_sub_iter,
                                             scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                             cv=self.cv, refit=self.scoring_metric, verbose=0,
                                             return_train_score=True)
            search_kpca.fit(self.X, self.y)
            search_grp = RandomizedSearchCV(pipe, param_grid[2], n_iter=n_sub_iter,
                                            scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                            cv=self.cv, refit=self.scoring_metric, verbose=0,
                                            return_train_score=True)
            search_grp.fit(self.X, self.y)
            search_fa = RandomizedSearchCV(pipe, param_grid[3], n_iter=n_sub_iter,
                                           scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                           cv=self.cv, refit=self.scoring_metric, verbose=0,
                                           return_train_score=True)
            search_fa.fit(self.X, self.y)
            search_k_best = RandomizedSearchCV(pipe, param_grid[4], n_iter=n_sub_iter,
                                               scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                               cv=self.cv, refit=self.scoring_metric, verbose=0,
                                               return_train_score=True)
            search_k_best.fit(self.X, self.y)

            best_scores = [search_pca.best_score_, search_kpca.best_score_, search_grp.best_score_,
                           search_fa.best_score_, search_k_best.best_score_]

            if min(best_scores) < 0:
                index = best_scores.index(min(best_scores))
            else:
                index = best_scores.index(max(best_scores))

            if index == 0:
                search = search_pca
            elif index == 1:
                search = search_kpca
            elif index == 2:
                search = search_grp
            elif index == 3:
                search = search_fa
            else:
                search = search_k_best
        else:
            raise Exception('Invalid value for search_type.')

        self.print_message('Finished algorithm = {0}.'.format(algorithm))
        self.print_message('Best score found = {0}'.format(search.best_score_))
        self.print_message('Best parameters found:')
        self.print_message(search.best_params_, pprint=True)

        return param_grid, search.best_estimator_

    def run(self):
        """
        Begins the optimization process.

        Returns
        ----------
        pipeline : object
            The best fit combination of transforms and models discovered during the
            optimization process.
        """
        self.print_message('Beginning optimization procedure...')
        t0 = time.time()

        if self.task not in ['classification', 'regression']:
            raise Exception('Invalid value provided for task.')

        if isinstance(self.algorithm, six.string_types):
            param_grid, best_model = self._run_single(self.algorithm)
            self.param_grids_[self.algorithm] = param_grid
            self.best_models_[self.algorithm] = best_model
            final_model = best_model
        elif isinstance(self.algorithm, list):
            for alg in self.algorithm:
                param_grid, best_model = self._run_single(alg)
                self.param_grids_[alg] = param_grid
                self.best_models_[alg] = best_model

            self.print_message('Completed algorithm optimization.')
            self.print_message('Initiating ensemble optimization...')

            model_tuples = []
            for key in self.best_models_:
                model_tuples.append((key, self.best_models_[key]))

            blender = Blender(models=model_tuples, scoring_metric=self.scoring_metric,
                              n_jobs=self.n_jobs, verbose=False)
            blender.build_ensemble(self.X, self.y, self.cv, self.task, retrain=True)
            self.print_message('Ensemble score = {0}'
                               .format(np.around(blender.test_score_, decimals=4)))
            final_model = copy.deepcopy(blender.ensemble_)
            ensemble_score = blender.test_score_

            self.print_message('Building stacked ensemble...')
            mask = [0 for _ in model_tuples]
            if self.task == 'classification':
                model_tuples.append(('stacker', ModelConfig.get_model(self.task, 'logistic')))
            else:
                model_tuples.append(('stacker', ModelConfig.get_model(self.task, 'ridge')))
            mask.append(1)

            blender = Blender(models=model_tuples, scoring_metric=self.scoring_metric,
                              n_jobs=self.n_jobs, verbose=False)
            blender.build_stacking_ensemble(self.X, self.y, self.cv, self.cv, mask, retrain=True)
            self.print_message('Stacking ensemble score = {0}'
                               .format(np.around(blender.test_score_, decimals=4)))
            if abs(blender.test_score_) > abs(ensemble_score):
                final_model = copy.deepcopy(blender.ensemble_)
        else:
            raise Exception('Invalid type provided for algorithm.')

        t1 = time.time()
        self.print_message('Optimization complete in {0:3f} s.'.format(t1 - t0))

        return final_model
