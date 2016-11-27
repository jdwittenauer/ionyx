from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from ionyx.datasets import *
from ionyx.ensemble import *
from ionyx.experiment import *
from ionyx.utils import *
from ionyx.visualization import *

# Object definitions
transforms = [StandardScaler()]
viz_transforms = [StandardScaler(), PCA(n_components=2, whiten=False)]
model = LogisticRegression()
log = Logger(path='log.txt', mode='replace')

# Load the data
data, X, y = load_forest_cover()

# Run some visualizations
visualize_feature_distributions(data)
visualize_correlations(data)
visualize_variable_relationships(data, ['Elevation'], category_vars=['Cover_Type'])
visualize_transforms(X, viz_transforms, y=y, model_type='classification')

# Train the model
model = train_model(X, y, model, library='sklearn', metric='accuracy', transforms=transforms, verbose=True, logger=log)

# Close the connection to the log file
log.close()
