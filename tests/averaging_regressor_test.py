import pprint as pp
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import LinearSVR
from ionyx.contrib import AveragingRegressor
from ionyx.datasets import DataSetLoader

print('Beginning averaging regressor test...')

data, X, y = DataSetLoader.load_property_inspection()
data = data.iloc[:1000, :]
X = X[:1000, :]
y = y[:1000]

estimators = [('ridge', Ridge()), ('lasso', Lasso()), ('svm', LinearSVR())]
ensemble = AveragingRegressor(estimators, weights=[1.0, 1.5, 2.0])
ensemble.fit(X, y)
print('Estimators list:')
pp.pprint(ensemble.estimators_)
print('Named estimators dict:')
pp.pprint(ensemble.named_estimators_)
print('Model 1 score = {0}'.format(mean_absolute_error(y, ensemble.estimators_[0].predict(X))))
print('Model 2 score = {0}'.format(mean_absolute_error(y, ensemble.estimators_[1].predict(X))))
print('Model 3 score = {0}'.format(mean_absolute_error(y, ensemble.estimators_[2].predict(X))))
print('Ensemble score = {0}'.format(mean_absolute_error(y, ensemble.predict(X))))

cv = KFold()
print('Cross-validation score = {0}'.format(cross_val_score(ensemble, X, y, cv=cv)))

param_grid = [
    {
        'ridge__alpha': [0.01, 0.1]
    }
]
grid = GridSearchCV(ensemble, param_grid=param_grid, cv=cv, return_train_score=True)
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results = results.sort_values(by='mean_test_score', ascending=False)
print('Grid search results:')
print(results)

print('Done.')
