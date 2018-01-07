import numpy as np
from ionyx.contrib import CategoryEncoder

print('Beginning category encoder test...')

X = np.array([['a', 'x', 1], ['b', 'x', 2], ['c', 'y', 3], ['a', 'y', 4]])
encoder = CategoryEncoder(categorical_features=[0, 1])
X_trans = encoder.fit_transform(X)
print('X:')
print(X)
print('X_trans:')
print(X_trans)

print('Done.')
