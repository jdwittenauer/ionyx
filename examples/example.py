from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
 
from ionyx.datasets import *
from ionyx.ensemble import *
from ionyx.experiment import *
from ionyx.utils import *
from ionyx.visualization import *
 
# Object definitions
transforms1 = [StandardScaler()]
transforms2 = [StandardScaler(), PCA(n_components=2, whiten=False)]
model = GradientBoostingClassifier()
log = Logger(path='log.txt')
 
# Load the data
data, X, y = load_forest_cover()
 
# Run some visualizations
visualize_feature_distributions(data)
visualize_correlations(data)
visualize_variable_relationships(data, ['Elevation'], category_vars=['Cover_Type'])
visualize_transforms(X, transforms2, y=y, model_type='classification')
 
# Train the model
model = train_model(X, y, model, library='sklearn', metric='f1', transforms=transforms1, verbose=True, logger=log)
visualize_feature_importance(model.feature_importances_, data.columns())
