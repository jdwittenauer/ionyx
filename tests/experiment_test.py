from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from ionyx import Experiment
from ionyx.datasets import DataSetLoader

_, X, y = DataSetLoader.load_property_inspection()
ridge = Ridge()
cv = KFold()
experiment = Experiment(package='sklearn', model=ridge, scoring_metric='neg_mean_squared_error',
                        verbose=True)
experiment.train_model(X, y)
experiment.cross_validate(X, y, cv)
experiment.learning_curve(X, y, cv)

param_grid = [
    {
        'alpha': [0.01, 0.1, 1.0]
    }
]
experiment.param_search(X, y, cv, param_grid, save_results_path='/home/john/temp/search.csv')
print(experiment.best_model_)

experiment.save_model('/home/john/temp/model.pkl')
experiment.load_model('/home/john/temp/model.pkl')
print(experiment.model)
