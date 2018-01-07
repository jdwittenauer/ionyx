import numpy as np
from keras.utils.np_utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from ionyx.contrib.keras_builder import KerasBuilder
from ionyx.datasets import DataSetLoader

print('Beginning keras builder test...')

data, X, y = DataSetLoader.load_forest_cover()
n_classes = len(np.unique(y)) + 1

model = KerasBuilder.build_dense_model(input_size=X.shape[1], output_size=n_classes,
                                       loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, to_categorical(y, n_classes))
score = model.evaluate(X, to_categorical(y, n_classes))
print('Model score = {0}'.format(score[1]))

estimator = KerasClassifier(build_fn=KerasBuilder.build_dense_model, input_size=X.shape[1],
                            output_size=n_classes, loss='categorical_crossentropy',
                            metrics=['accuracy'])
estimator.fit(X, to_categorical(y, n_classes))
score = estimator.score(X, to_categorical(y, n_classes))
print('Estimator score = {0}'.format(score))

print('Done.')
