import numpy as np
from ionyx.contrib import TimeSeriesSplit

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 1, 2, 3, 4, 5, 6, 7])

tscv = TimeSeriesSplit(n_splits=3)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

tscv = TimeSeriesSplit(n_splits=3, min_train_size=4)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

tscv = TimeSeriesSplit(n_splits=3, max_train_size=4)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)

tscv = TimeSeriesSplit(n_splits=3, max_train_size=4, min_train_size=4)
print(tscv)
for train_index, test_index in tscv.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
