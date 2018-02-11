import os


class SuppressOutput(object):
    """
    A context manager for doing a "deep suppression" of stdout and stderr in Python, i.e.
    will suppress all print, even if the print originates in a compiled C/Fortran
    sub-function.  This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has exited.  Use
    in a "with" block for temporary suppression of specific sections of code.
    """
    def __init__(self):
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for _ in range(2)]
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        for fd in self.null_fds + self.save_fds:
            os.close(fd)
