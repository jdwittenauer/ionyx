from sklearn.decomposition import PCA
from ionyx.datasets import DataSetLoader
from ionyx import Visualizer

print('Beginning visualizer test...')

data, _, _ = DataSetLoader.load_property_inspection()
viz = Visualizer(data)
viz.feature_distributions()
viz.feature_correlations()
viz.variable_relationship(['T1_V1', 'T1_V2'], category_vars=None)
viz.variable_relationship(['T1_V1', 'T1_V2', 'T2_V1'], category_vars=None)
viz.variable_relationship(['T2_V1'], category_vars=['T1_V4'])
viz.variable_relationship(['T2_V1'], category_vars=['T1_V4', 'T1_V7'])
viz.variable_relationship(['T1_V1', 'T2_V1'], category_vars=['T1_V7'])

data, _, _, _ = DataSetLoader.load_bike_sharing()
viz = Visualizer(data)
viz.sequential_relationships()

data, _, _ = DataSetLoader.load_otto_group()
X_cols = data.columns[1:].tolist()
y_col = data.columns[0]
viz = Visualizer(data)
pca = PCA()
viz.transform(pca, X_columns=X_cols)
viz.transform(pca, X_columns=X_cols, y_column=y_col, task='classification')
viz.transform(pca, X_columns=X_cols, y_column=y_col, task='regression')

data, _, _ = DataSetLoader.load_otto_group()
data = data.iloc[:10000, :30]
X_cols = data.columns[1:].tolist()
y_col = data.columns[0]
viz = Visualizer(data)
viz.feature_importance(X_columns=X_cols, y_column=y_col, average=True, task='classification')
viz.partial_dependence(X_columns=X_cols, y_column=y_col, var_column='feat_15', average=True,
                       task='classification')

print('Done.')
