class Logger(object):
    """
    Wrapper class that permits writing to a log file as well as printing to the console at the same time using
    a single convenience method.  Instantiating this class opens a file connection that must be closed.

    Parameters
    ----------
    path : string
        The location of the text file to open or create.  Will append if existing file is specified.
    """
    def __init__(self, path):
        self.log = open(path, 'a')

    def write(self, message):
        """
        Write the message to a log file and print to the console at the same time.

        Parameters
        ----------
        message : string
            Text message to record and print.
        """
        self.log.write(message)
        print(message)

    def close(self):
        """
        Close the connection to the underlying file.
        """
        self.log.close()

    def __repr__(self):
        """
        Overrides the method that prints a string representation of the object.
        """
        return '%s' % self.__class__.__name__
