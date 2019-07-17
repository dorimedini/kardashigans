import logging


class Verbose(object):
    """ Inherit this class to call self._print and get line number etc."""
    def __init__(self, name=None, verbose=False):
        self._verbose = verbose
        self._name = name if name else self.__class__.__name__
        self._init_logging()

    def _init_logging(self):
        self.logger = logging.getLogger(self._name)
        level = logging.DEBUG if self._verbose else logging.WARNING
        self.logger.setLevel(level=level)

    def _print(self, *args, **kwargs):
        if self._verbose is False:
            return
        self.logger.debug(args[0], *args[1:], **kwargs)
