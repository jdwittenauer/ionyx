import os
import pandas as pd
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
from ..contrib import CategoryEncoder


class DataSetLoader(object):
    """
    Provides a number of pre-staged data sets to load into memory.
    """
    def __init__(self):
        pass

    @staticmethod
    def load_bike_sharing():
        """
        Loads and returns the data set from Kaggle's Bike Sharing Demand competition.
        Link: https://www.kaggle.com/c/bike-sharing-demand

        Returns
        ----------
        data : array-like
            Pandas data frame containing the entire data set.

        X : array-like
            Training input samples.

        y1 : array-like
            First variable target values.

        y2 : array-like
            Second variable target values.
        """
        file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'bike_sharing.zip')
        z = ZipFile(file_location)
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

    @staticmethod
    def load_forest_cover():
        """
        Loads and returns the data set from Kaggle's Forest Cover Type Prediction competition.
        Link: https://www.kaggle.com/c/forest-cover-type-prediction

        Returns
        ----------
        data : array-like
            Pandas data frame containing the entire data set.

        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'forest_cover.zip')
        z = ZipFile(file_location)
        data = pd.read_csv(z.open('train.csv'))
        data = data.set_index('Id')

        # move the label to the first position
        cols = data.columns.tolist()
        cols = cols[-1:] + cols[0:-1]
        data = data[cols]

        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        return data, X, y

    @staticmethod
    def load_otto_group():
        """
        Loads and returns the data set from Kaggle's Otto Group Product Classification competition.
        Link: https://www.kaggle.com/c/otto-group-product-classification-challenge

        Returns
        ----------
        data : array-like
            Pandas data frame containing the entire data set.

        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'otto_group.zip')
        z = ZipFile(file_location)
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

    @staticmethod
    def load_property_inspection():
        """
        Loads and returns the data set from Kaggle's Property Inspection Prediction competition.
        Link: https://www.kaggle.com/c/liberty-mutual-group-property-inspection-prediction

        Returns
        ----------
        data : array-like
            Pandas data frame containing the entire data set.

        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'property_inspection.zip')
        z = ZipFile(file_location)
        data = pd.read_csv(z.open('train.csv'))
        data = data.set_index('Id')

        X = data.iloc[:, 1:].values
        y = data.iloc[:, 0].values

        # transform the categorical variables from strings to integers
        encoder = CategoryEncoder()
        X = encoder.fit_transform(X)

        return data, X, y

    @staticmethod
    def load_time_series():
        """
        Loads and returns a generic time series data set.

        Returns
        ----------
        data : array-like
            Pandas data frame containing the entire data set.

        X : array-like
            Training input samples.

        y : array-like
            Target values.
        """
        file_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'time_series.zip')
        z = ZipFile(file_location)
        data = pd.read_csv(z.open('train.csv'))

        X = data['ds'].values
        y = data['y'].values

        return data, X, y
