import pickle
import pandas as pd
from sklearn.metrics import *


def create_class(import_path, module_name, class_name, *params):
    """
    Returns an instantiated class for the given string descriptors.

    Parameters
    ----------
    import_path : string
        The path to the module.

    module_name : string
        The module name.

    class_name : string
        The class name.

    params : array-like
        Any fields required to instantiate the class.

    Returns
    ----------
    instance : object
        An instance of the class.
    """
    p = __import__(import_path)
    m = getattr(p, module_name)
    c = getattr(m, class_name)
    instance = c(*params)

    return instance


def load_csv_data(filename, dtype=None, index=None, convert_to_date=False):
    """
    Load a csv data file into a data frame.  This function wraps Pandas' read_csv function with logic to set the
    index for the data frame and convert fields to date types if necessary.

    Parameters
    ----------
    filename : string
        Location of the file to read.

    dtype : array-like, optional, default None
        List of column types to set for the data frame.

    index : string, array-like, optional, default None
        String or list of strings specifying the columns to use as an index.

    convert_to_date : boolean, optional, default False
        Boolean indicating if the index consists of date fields.

    Returns
    ----------
    data : array-like
        Data frame containing data from the input file.
    """
    data = pd.read_csv(filename, sep=',', dtype=dtype)

    if index is not None:
        if convert_to_date:
            if type(index) is str:
                data[index] = data[index].convert_objects(convert_dates='coerce')
            else:
                for key in index:
                    data[key] = data[key].convert_objects(convert_dates='coerce')

        data = data.set_index(index)

    print('Data file ' + filename + ' loaded successfully.')

    return data


def load_model(filename):
    """
    Load a previously trained model from disk.

    Parameters
    ----------
    filename : string
        Location of the file to read.

    Returns
    ----------
    model : object
        An object in memory that represents a fitted model.
    """
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    return model


def save_model(model, filename):
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : object
        An object in memory that represents a fitted model.

    filename : string
        Location of the file to write.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()


def fit_transforms(X, y, transforms):
    """
    Fits new transformations from a data set.

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    transforms : array-like
        List of objects with a fit_transform function that accepts two parameters.

    Returns
    ----------
    transforms : array-like
        List of transform objects after calling fit_transform on the input data.
    """
    for i, trans in enumerate(transforms):
        if trans is not None:
            X = trans.fit_transform(X, y)
        transforms[i] = trans

    return transforms


def apply_transforms(X, transforms):
    """
    Applies pre-computed transformations to a data set.

    Parameters
    ----------
    X : array-like
        Training input samples.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    Returns
    ----------
    X : array-like
        Training input samples after iteratively applying each transform to the data.
    """
    for trans in transforms:
        if trans is not None:
            X = trans.transform(X)

    return X


def score(y, y_pred, metric):
    """
    Calculates a score for the given predictions using the provided metric.

    Parameters
    ----------
    y : array-like
        Target values.

    y_pred : array-like
        Predicted target values.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc'}
        Scoring metric.

    Returns
    ----------
    score : float
        Calculated score between the target and predicted target values.
    """
    y = y.ravel()
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

    Parameters
    ----------
    X : array-like
        Training input samples.

    y : array-like
        Target values.

    model : object
        An object in memory that represents a fitted model.

    metric : {'accuracy', 'f1', 'log_loss', 'mean_absolute_error', 'mean_squared_error', 'r2', 'roc_auc', None}
        Scoring metric.

    Returns
    ----------
    score : float
        Calculated score between the target and predicted target values.
    """
    if metric is not None:
        y_pred = model.predict(X)
        return score(y, y_pred, metric)
    else:
        return model.score(X, y)
