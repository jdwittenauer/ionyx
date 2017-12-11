import pandas as pd


class Utils(object):
    """
    Utility class with miscellaneous useful independent functions.
    """
    def __init__(self):
        pass

    @staticmethod
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

    @staticmethod
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

        print_message('Data file {0} loaded successfully.'.format(str(filename)), verbose, logger)

        return data

    @staticmethod
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

        print_message('SQL query completed successfully.', verbose, logger)

        return data
