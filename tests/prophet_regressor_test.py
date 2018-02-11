import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from ionyx.contrib import ProphetRegressor
from ionyx.contrib import SuppressOutput
from ionyx.contrib import TimeSeriesSplit
from ionyx.datasets import DataSetLoader

print('Beginning prophet regressor test...')

data, X, y = DataSetLoader.load_time_series()
prophet = ProphetRegressor(n_changepoints=0)
with SuppressOutput():
    prophet.fit(X, y)
print('Model score = {0}'.format(mean_absolute_error(y, prophet.predict(X))))

cv = TimeSeriesSplit(n_splits=3)
with SuppressOutput():
    score = cross_val_score(prophet, X, y, cv=cv)
print('Cross-validation score = {0}'.format(score))

param_grid = [
    {
        'n_changepoints': [0, 25]
    }
]
grid = GridSearchCV(prophet, param_grid=param_grid, cv=cv, return_train_score=True)
with SuppressOutput():
    grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results = results.sort_values(by='mean_test_score', ascending=False)
print('Grid search results:')
print(results)

print('Done.')
