from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import shuffle
from ionyx import Blender
from ionyx.datasets import DataSetLoader

_, X, y = DataSetLoader.load_otto_group()
X, y = shuffle(X, y, n_samples=10000)
X = X[:, :30]
models = [('logistic', LogisticRegression()), ('svc', SVC()), ('decision_tree', DecisionTreeClassifier())]
cv = KFold()
blender = Blender(models=models, scoring_metric='neg_mean_squared_error')
blender.build_ensemble(X, y, cv, retrain=True)
print(blender.ensemble_)
