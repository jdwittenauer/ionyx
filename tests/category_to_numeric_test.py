import numpy as np
from ionyx.contrib import CategoryToNumeric

print('Beginning category to numeric test...')

X = np.array([[1.], [1.], [1.], [1.], [2.], [2.], [2.], [2.]])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8])

encoder = CategoryToNumeric(categorical_features=[0])
X_trans = encoder.fit_transform(X, y)
print('X:')
print(X)
print('X_trans:')
print(X_trans)

print('Done.')
