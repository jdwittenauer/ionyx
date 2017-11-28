class Experiment(object):
    def __init__(self, model, scoring_metric, verbose=False, logger=None):
        self.model = model
        self.scoring_metric = scoring_metric
        self.verbose = verbose
        self.logger = logger

        self.best_model_ = None

    def train_model(self, X, y):
        pass

    def cross_validate(self, X, y, cv):
        pass

    def param_search(self, X, y, grid):
        pass

    def load_model(self, path):
        pass

    def save_model(self, path):
        pass

    def score(self, y, y_pred):
        pass

    def predict_score(self, X, y):
        pass

    def __repr__(self):
        return '%s' % self.__class__.__name__
