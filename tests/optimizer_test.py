from sklearn.model_selection import KFold
from ionyx import Optimizer
from ionyx.datasets import DataSetLoader

print('Beginning experiment test...')

_, X, y = DataSetLoader.load_forest_cover()
X = X[:5000, :20]
y = y[:5000]
cv = KFold()
opt = Optimizer(X, y, cv, algorithm='bayes', scoring_metric='accuracy',
                task='classification', search_type='grid')
pipe = opt.run()
print(pipe)

_, X, y = DataSetLoader.load_property_inspection()
X = X[:5000, :20]
y = y[:5000]
cv = KFold()
opt = Optimizer(X, y, cv, algorithm='ridge', scoring_metric='neg_mean_squared_error',
                task='regression', search_type='random', n_iter=10)
pipe = opt.run()
print(pipe)

print('Done.')
