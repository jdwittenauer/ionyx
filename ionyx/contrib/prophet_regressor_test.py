import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, cross_val_score
from ionyx.contrib.prophet_regressor import ProphetRegressor
from ionyx.contrib.time_series_split import TimeSeriesSplit

path = os.path.join(os.getcwd() + '\\ionyx\\contrib', 'prophet_regressor_data.csv')
data = pd.read_csv(path, parse_dates=['ds'])
X = data['ds'].values
y = data['y'].values

prophet = ProphetRegressor(n_changepoints=0)
prophet.fit(X, y)
y_pred = prophet.predict(X)
print(mean_absolute_error(y, y_pred))

cv = TimeSeriesSplit(n_splits=3)
print(np.mean(cross_val_score(prophet, X, y, cv=cv)))

param_grid = [
    {
        'n_changepoints': [0, 25],
        'changepoint_prior_scale': [0.005, 0.05]
    }
]
grid = GridSearchCV(prophet, param_grid=param_grid, cv=cv, return_train_score=True)
grid.fit(X, y)
results = pd.DataFrame(grid.cv_results_)
results = results.sort_values(by='mean_test_score', ascending=False)
print(results)
