import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from ionyx import Optimizer
from ionyx.datasets import DataSetLoader

print('Beginning experiment test...')

_, X, y = DataSetLoader.load_forest_cover()
X, y = shuffle(X, y, n_samples=5000)
X = X[:, :20].astype(np.float64)
cv = KFold()
opt = Optimizer(X, y, cv, algorithm='bayes', scoring_metric='accuracy',
                task='classification', search_type='grid')
pipe = opt.run()
print(pipe)

_, X, y = DataSetLoader.load_property_inspection()
X, y = shuffle(X, y, n_samples=5000)
X = X[:, :20].astype(np.float64)
cv = KFold()
opt = Optimizer(X, y, cv, algorithm='ridge', scoring_metric='neg_mean_squared_error',
                task='regression', search_type='random', n_iter=10)
pipe = opt.run()
print(pipe)

_, X, y = DataSetLoader.load_forest_cover()
X, y = shuffle(X, y, n_samples=5000)
X = X[:, :20].astype(np.float64)
cv = KFold()
algorithms = ['bayes', 'logistic', 'random_forest']
opt = Optimizer(X, y, cv, algorithm=algorithms, scoring_metric='accuracy',
                task='classification', search_type='random', n_iter=10)
ensemble = opt.run()
print(ensemble)

print('Done.')
