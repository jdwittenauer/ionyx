class Logger(object):
    """
    Wrapper class that enables writing to a log file.  Instantiating this class opens
    a file connection that must be closed.

    Parameters
    ----------
    path : string
        The location of the text file to open or create.

    mode : {'append', 'replace'}, optional, default 'replace'
        Specifies whether to append or replace if file already exists.
    """
    def __init__(self, path, mode='replace'):
        if mode == 'append':
            self.log = open(path, 'a')
        elif mode == 'replace':
            self.log = open(path, 'w')
        else:
            raise Exception('Mode not supported.')

    def write(self, message):
        """
        Write the message to a log file.

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
