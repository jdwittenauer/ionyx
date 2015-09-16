import os
import pandas as pd
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder

from ..utils import CategoryEncoder


def load_bike_sharing():
    z = ZipFile(os.getcwd() + '/data/bike_sharing.zip')
    data = pd.read_csv(z.open('train.csv'))
    data['datetime'] = data['datetime'].convert_objects(convert_dates='coerce')
    data = data.set_index('datetime')

    # drop the total count label and move the registered/casual counts to the front
    num_features = len(data.columns) - 3
    cols = data.columns.tolist()
    cols = cols[-3:-1] + cols[0:num_features]
    data = data[cols]

    X = data.iloc[:, 2:].values
    y1 = data.iloc[:, 0].values
    y2 = data.iloc[:, 1].values

    return data, X, y1, y2


def load_forest_cover():
    z = ZipFile(os.getcwd() + '/data/forest_cover.zip')
    data = pd.read_csv(z.open('train.csv'))
    data = data.set_index('Id')

    # move the label to the first position
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[0:-1]
    data = data[cols]

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    return data, X, y


def load_otto_group():
    z = ZipFile(os.getcwd() + '/data/otto_group.zip')
    data = pd.read_csv(z.open('train.csv'))
    data = data.set_index('id')

    # move the label to the first position
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[0:-1]
    data = data[cols]

    X = data.iloc[:, 1:].values

    y = data.iloc[:, 0].values

    # transform the labels from strings to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    return data, X, y


def load_property_inspection():
    z = ZipFile(os.getcwd() + '/data/property_inspection.zip')
    data = pd.read_csv(z.open('train.csv'))
    data = data.set_index('Id')

    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # transform the categorical variables from strings to integers
    encoder = CategoryEncoder()
    X = encoder.fit_transform(X)

    return data, X, y
