import datetime
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


def print_status_message(message, verbose=False, logger=None):
    """
    Wrapper function that encapsulates the logic of figuring out how to handle status messages.

    Parameters
    ----------
    message : string
        Generic text message.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.
    """
    if verbose:
        now = datetime.datetime.now().replace(microsecond=0).isoformat(' ')
        print('(' + now + ') ' + message)

        if logger is not None:
            logger.write('(' + now + ') ' + message + '\n')


def load_csv_data(filename, dtype=None, index=None, convert_to_date=False, verbose=False, logger=None):
    """
    Load a csv data file into a data frame.  This function wraps the Pandas read_csv function with logic
    to set the index for the data frame and convert fields to date types if necessary.

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

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

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

    print_status_message('Data file {0} loaded successfully.'.format(str(filename)), verbose, logger)

    return data


def load_sql_data(engine, table=None, query=None, index=None, params=None,
                  date_columns=None, verbose=False, logger=None):
    """
    Reads SQL data using the specified table or query and returns a data frame with the results.  This function
    wraps Pandas' read_sql function with logic to allow for a table name to be specified instead of a query.

    Parameters
    ----------
    engine : object
        A SQLAlchemy engine with a connection to the database to read from.

    table : string, optional, default None
        Name of the table to read from if reading entire table contents.  Specify either table or query parameter.

    query : string, optional, default None
        SQL query to run against the database.  Specify either table or query parameter.

    index : string or array-like, optional, default None
        Specify either the column or list of columns to set as the index of the data frame.

    params: array-like, optional, default None
        List of input parameters for the database to evaluate with the SQL query.

    date_columns : array-like, optional, default None
        List of column names to parse as dates in the data frame.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    data : array-like
        Data frame containing the results of the query.
    """
    if query is not None:
        data = pd.read_sql(query, engine, index_col=index, params=params, parse_dates=date_columns)
    else:
        data = pd.read_sql('SELECT * FROM ' + table, engine, index_col=index,
                           params=params, parse_dates=date_columns)

    print_status_message('SQL query completed successfully.', verbose, logger)

    return data


def load_model(filename, verbose=False, logger=None):
    """
    Load a previously trained model from disk.

    Parameters
    ----------
    filename : string
        Location of the file to read.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    model : object
        An object in memory that represents a fitted model.
    """
    model_file = open(filename, 'rb')
    model = pickle.load(model_file)
    model_file.close()

    print_status_message('Loaded model from {0} into memory.'.format(str(filename)), verbose, logger)

    return model


def save_model(model, filename, verbose=False, logger=None):
    """
    Persist a trained model to disk.

    Parameters
    ----------
    model : object
        An object in memory that represents a fitted model.

    filename : string
        Location of the file to write.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.
    """
    model_file = open(filename, 'wb')
    pickle.dump(model, model_file)
    model_file.close()

    print_status_message('Saved model to disk at {0}.'.format(str(filename)), verbose, logger)


def fit_transforms(X, y, transforms, verbose=False, logger=None):
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

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    transforms : array-like
        List of transform objects after calling fit_transform on the input data.
    """
    print_status_message('Fitting transforms...', verbose, logger)
    for i, trans in enumerate(transforms):
        if trans is not None:
            X = trans.fit_transform(X, y)
        transforms[i] = trans

    print_status_message('Transform fitting complete.', verbose, logger)

    return transforms


def apply_transforms(X, transforms, verbose=False, logger=None):
    """
    Applies pre-computed transformations to a data set.

    Parameters
    ----------
    X : array-like
        Training input samples.

    transforms : array-like
        List of objects with a transform function that accepts one parameter.

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    X : array-like
        Training input samples after iteratively applying each transform to the data.
    """
    print_status_message('Applying transforms...', verbose, logger)
    for trans in transforms:
        if trans is not None:
            X = trans.transform(X)

    print_status_message('Transform application complete.', verbose, logger)

    return X


def score(y, y_pred, metric, verbose=False, logger=None):
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

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    s : float
        Calculated score between the target and predicted target values.
    """
    s = None
    y = y.ravel()
    y_pred = y_pred.ravel()
    assert y.shape == y_pred.shape, 'Shape of y and y_pred do not match.'

    if metric == 'accuracy':
        s = accuracy_score(y, y_pred)
    elif metric == 'f1':
        s = f1_score(y, y_pred)
    elif metric == 'log_loss':
        s = log_loss(y, y_pred)
    elif metric == 'mean_absolute_error':
        s = mean_absolute_error(y, y_pred)
    elif metric == 'mean_squared_error':
        s = mean_squared_error(y, y_pred)
    elif metric == 'r2':
        s = r2_score(y, y_pred)
    elif metric == 'roc_auc':
        s = roc_auc_score(y, y_pred)

    if s is not None:
        print_status_message('Score = {0}'.format(str(round(s, 6))), verbose, logger)
        return s
    else:
        raise Exception('Invalid metric was provided: ' + str(metric))


def predict_score(X, y, model, metric, verbose=False, logger=None):
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

    verbose : boolean, optional, default False
        Prints status messages to the console if enabled.

    logger : object, optional, default None
        Instance of a class that can log messages to an output file.

    Returns
    ----------
    s : float
        Calculated score between the target and predicted target values.
    """
    if metric is not None:
        y_pred = model.predict(X)
        return score(y, y_pred, metric, verbose, logger)
    else:
        s = model.score(X, y)
        print_status_message('Score = {0}'.format(str(round(s, 6))), verbose, logger)
        return s
