from sklearn.decomposition import PCA
from ionyx.datasets import DataSetLoader
from ionyx import Visualizer

data, _, _ = DataSetLoader.load_property_inspection()
viz = Visualizer(data)
viz.feature_distributions()
viz.correlations()
viz.variable_relationship(['T1_V1', 'T1_V2'], category_vars=None)
viz.variable_relationship(['T1_V1', 'T1_V2', 'T2_V1'], category_vars=None)
viz.variable_relationship(['T2_V1'], category_vars=['T1_V4'])
viz.variable_relationship(['T2_V1'], category_vars=['T1_V4', 'T1_V7'])
viz.variable_relationship(['T1_V1', 'T2_V1'], category_vars=['T1_V7'])

data, _, _, _ = DataSetLoader.load_bike_sharing()
viz = Visualizer(data)
viz.sequential_relationships()

data, X, y = DataSetLoader.load_otto_group()
viz = Visualizer(data)
pca = PCA()
viz.transform(pca, X_columns=data.columns[1:])
viz.transform(pca, X_columns=data.columns[1:], y_column=data.columns[0], supervision_task='classification')
viz.transform(pca, X_columns=data.columns[1:], y_column=data.columns[0], supervision_task='regression')
