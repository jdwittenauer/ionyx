import datetime
import pprint as pp


class PrintMessageMixin(object):
    def __init__(self, verbose=True, logger=None):
        self.verbose = verbose
        self.logger = logger

    def print_message(self, message, pprint=False):
        """
        Optionally print a message to the console and/or log to a file.

        Parameters
        ----------
        message : string
            Generic text message.

        pprint : boolean, optional, default False
            Enables stylistic formatting.
        """
        now = datetime.datetime.now().replace(microsecond=0).isoformat(' ')
        if self.verbose:
            if pprint:
                pp.pprint(message)
            else:
                print('(' + now + ') ' + message)
        if self.logger:
            if pprint:
                self.logger.write(pp.pformat(message))
            else:
                self.logger.write('(' + now + ') ' + message)
