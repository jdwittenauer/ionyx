import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .contrib.model_config import ModelConfig
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
                 'keras', 'prophet'}
        Model type to optimize.  Source package will be selected automatically based on
        the algorithm provided.

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
    param_grid_ : list
        The hyper-parameter grid constructed by the algorithm to conduct a search.
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
        self.param_grid_ = None
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

    def run(self):
        """
        Begins the optimization process.

        Returns
        ----------
        pipeline : object
            The best fit transforms and model discovered during the optimization process.
        """
        self.print_message('Beginning optimization procedure...')
        t0 = time.time()

        if self.algorithm not in ['bayes', 'logistic', 'ridge', 'elastic_net', 'linear_svm', 'svm',
                                  'random_forest', 'extra_trees', 'gradient_boosting', 'xgboost',
                                  'keras', 'prophet']:
            raise Exception('Invalid value for algorithm.')

        if self.task not in ['classification', 'regression']:
            raise Exception('Invalid value for task.')

        scaler = StandardScaler()
        transform = ModelConfig.get_transform('pca')
        model = ModelConfig.get_model(self.task, self.algorithm)
        pipe = Pipeline([('scaler', scaler), ('transform', transform), ('model', model)])
        n_features = self.X.shape[1]

        if self.algorithm == 'keras':
            model.set_params(verbose=0)

        self.print_message('Building parameter grid...')
        self.param_grid_ = [
            ModelConfig.transform_search_params(self.search_type, 'pca', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'kpca', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'grp', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'fa', n_features, 'transform'),
            ModelConfig.transform_search_params(self.search_type, 'k_best', n_features, 'transform')
        ]
        model_grid = ModelConfig.model_search_params(self.search_type, self.task, self.algorithm, 'model')
        self.param_grid_ = [_combine_dict(x, model_grid) for x in self.param_grid_]

        self.print_message('Initiating search...')
        if self.search_type == 'grid':
            search = GridSearchCV(pipe, param_grid=self.param_grid_, scoring=self.scoring_metric,
                                  n_jobs=self.n_jobs, cv=self.cv, refit=self.scoring_metric,
                                  verbose=0, return_train_score=True)
        elif self.search_type == 'random':
            # TODO - won't work, doesn't handle list of dict for param_grid
            search = RandomizedSearchCV(pipe, self.param_grid_, n_iter=self.n_iter,
                                        scoring=self.scoring_metric, n_jobs=self.n_jobs,
                                        cv=self.cv, refit=self.scoring_metric, verbose=0,
                                        return_train_score=True)
        else:
            raise Exception('Invalid value for search_type.')

        search.fit(self.X, self.y)

        t1 = time.time()
        self.print_message('Optimization complete in {0:3f} s.'.format(t1 - t0))
        self.print_message('Best score found = {0}'.format(search.best_score_))
        self.print_message('Best parameters found:')
        self.print_message(search.best_params_, pprint=True)

        return search.best_estimator_
