import time
from .print_message import PrintMessageMixin


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
        self.param_grid = param_grid
        self.n_jobs = n_jobs
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

        if self.search_type not in ['grid', 'random']:
            raise Exception('Invalid value for search_type.')

        t1 = time.time()
        self.print_message('Optimization complete in {0:3f} s.'.format(t1 - t0))

        return None
