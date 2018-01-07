from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from keras.wrappers.scikit_learn import KerasRegressor
from ionyx import Experiment
from ionyx.contrib.keras_builder import KerasBuilder
from ionyx.datasets import DataSetLoader

print('Beginning experiment test...')

data, _, _ = DataSetLoader.load_forest_cover()
X_cols = data.columns[1:].tolist()
y_col = data.columns[0]
logistic = LogisticRegression()
cv = KFold()
experiment = Experiment(package='sklearn', model=logistic, scoring_metric='accuracy',
                        verbose=True, data=data, X_cols=X_cols, y_col=y_col, cv=cv)
experiment.train_model()
experiment.cross_validate()
experiment.learning_curve()
param_grid = [
    {
        'alpha': [0.01, 0.1, 1.0]
    }
]
experiment.param_search(param_grid, save_results_path='/home/john/temp/search.csv')
print(experiment.best_model_)
experiment.save_model('/home/john/temp/model.pkl')
experiment.load_model('/home/john/temp/model.pkl')
print(experiment.model)

_, X, y = DataSetLoader.load_property_inspection()
xgb = XGBRegressor()
cv = KFold()
experiment = Experiment(package='xgboost', model=xgb, scoring_metric='neg_mean_squared_error',
                        eval_metric='rmse', verbose=True)
experiment.train_model(X, y, validate=True, early_stopping=True, early_stopping_rounds=5,
                       plot_eval_history=True)
experiment.cross_validate(X, y, cv)
experiment.save_model('/home/john/temp/model.pkl')
experiment.load_model('/home/john/temp/model.pkl')
print(experiment.model)

_, X, y = DataSetLoader.load_property_inspection()
nn = KerasRegressor(build_fn=KerasBuilder.build_dense_model, input_size=X.shape[1], output_size=1,
                    loss='mean_squared_error', metrics=['mse'], batch_size=32, epochs=5)
cv = KFold()
experiment = Experiment(package='keras', model=nn, scoring_metric='neg_mean_squared_error',
                        verbose=True)
experiment.train_model(X, y, validate=True, early_stopping=True, early_stopping_rounds=2,
                       plot_eval_history=True)
experiment.cross_validate(X, y, cv)
experiment.save_model('/home/john/temp/model.h5')
experiment.load_model('/home/john/temp/model.h5')
print(experiment.model)

print('Done.')
