import pickle
from sklearn.metrics import *

def load_model(filename):
    """
    Load a previously training model from disk.
    """
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    return model


def save_model(model, filename):
    """
    Persist a trained model to disk.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def fit_transforms(X, y, transforms):
    """
    Fits new transformations from a data set.
    """
    for i, trans in enumerate(transforms):
        if trans is not None:
            X = trans.fit_transform(X, y)
        transforms[i] = trans

    return transforms


def apply_transforms(X, transforms):
    """
    Applies pre-computed transformations to a data set.
    """
    for trans in transforms:
        if trans is not None:
            X = trans.transform(X)

    return X


def score(y, y_pred, metric):
    """
    Calculates a score for the given predictions using the provided metric.
    """
    y_pred = y_pred.ravel()
    assert y.shape == y_pred.shape

    if metric == 'accuracy':
        return accuracy_score(y, y_pred)
    elif metric == 'f1':
        return f1_score(y, y_pred)
    elif metric == 'log_loss':
        return log_loss(y, y_pred)
    elif metric == 'mean_absolute_error':
        return mean_absolute_error(y, y_pred)
    elif metric == 'mean_squared_error':
        return mean_squared_error(y, y_pred)
    elif metric == 'r2':
        return r2_score(y, y_pred)
    elif metric == 'roc_auc':
        return roc_auc_score(y, y_pred)


def predict_score(X, y, model, metric):
    """
    Predicts and scores the model's performance and returns the result.
    """
    if metric is not None:
        y_pred = model.predict(X)
        return score(y, y_pred, metric)
    else:
        return model.score(X, y)
