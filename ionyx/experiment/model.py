import time
from xgboost import *
from keras.callbacks import *
from keras.layers.core import *
from keras.layers.normalization import *
from keras.layers.advanced_activations import *
from keras.models import *
from keras.optimizers import *

from ..utils import fit_transforms, apply_transforms, predict_score


def train_model(X, y, model, metric, transforms):
    """
    Trains a new model using the training data.
    """
    t0 = time.time()
    transforms = fit_transforms(X, y, transforms)
    X = apply_transforms(X, transforms)
    model.fit(X, y)
    t1 = time.time()
    print('Model trained in {0:3f} s.'.format(t1 - t0))

    print('Model hyper-parameters:')
    print(model.get_params())

    print('Calculating training score...')
    model_score = predict_score(X, y, model, metric)
    print('Training score ='), model_score

    return model
