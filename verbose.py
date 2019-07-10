from inspect import getframeinfo, stack


class Verbose(object):
    """ Inherit this class to call self._print and get line number etc."""
    def __init__(self, verbose=False):
        self._verbose = verbose

    def _print(self, *args, **kwargs):
        caller = getframeinfo(stack()[1][0])
        if self._verbose:
            print("%s:%d - %s" % (caller.filename, caller.lineno, args[0]), *args[1:], **kwargs)
