from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from ionyx import Blender
from ionyx.datasets import DataSetLoader

print('Beginning blender test...')

_, X, y = DataSetLoader.load_otto_group()
X, y = shuffle(X, y, n_samples=10000)
X = X[:, :30]
models = [('logistic', LogisticRegression()), ('svc', SVC()), ('decision_tree', DecisionTreeClassifier())]
cv = KFold()
blender = Blender(models=models, scoring_metric='neg_mean_squared_error')
blender.build_ensemble(X, y, cv, retrain=True)
print(blender.ensemble_)

_, X, y = DataSetLoader.load_otto_group()
X, y = shuffle(X, y, n_samples=10000)
X = X[:, :30]
models = [('logistic', LogisticRegression()), ('svc', SVC()), ('decision_tree', DecisionTreeClassifier()),
          ('stacker', LogisticRegression())]
layer_mask = [0, 0, 0, 1]
cv = KFold()
layer_cv = KFold()
blender = Blender(models=models, scoring_metric='neg_mean_squared_error')
blender.build_stacking_ensemble(X, y, cv, layer_cv, layer_mask, retrain=True)
print(blender.ensemble_)

_, X, y = DataSetLoader.load_otto_group()
X, y = shuffle(X, y, n_samples=10000)
X = X[:, :30]
models = [('logistic', LogisticRegression()), ('svc', SVC()), ('decision_tree', DecisionTreeClassifier())]
cv = KFold()
blender = Blender(models=models, scoring_metric='neg_mean_squared_error')
blender.model_correlations(X, y, cv)

print('Done.')
