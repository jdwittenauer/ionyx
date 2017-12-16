class Logger(object):
    """
    Wrapper class that permits writing to a log file as well as printing to the console at the same time using
    a single convenience method.  Instantiating this class opens a file connection that must be closed.

    Parameters
    ----------
    path : string
        The location of the text file to open or create.

    mode : { 'append', 'replace' }, optional, default 'replace'
        Specifies whether to append or replace if file already exists.
    """
    def __init__(self, path, mode='replace'):
        if mode == 'append':
            self.log = open(path, 'a')
        else:
            self.log = open(path, 'w')

    def write(self, message):
        """
        Write the message to a log file and print to the console at the same time.

        Parameters
        ----------
        message : string
            Text message to record and print.
        """
        self.log.write(message + '\n')

    def close(self):
        """
        Close the connection to the underlying file.
        """
        self.log.close()
