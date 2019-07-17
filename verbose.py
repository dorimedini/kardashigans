from datetime import datetime
from inspect import getframeinfo, stack
import os
import pytz


class Verbose(object):
    """ Inherit this class to call self._print and get line number etc."""
    def __init__(self, verbose=False):
        self._verbose = verbose

    def _print(self, *args, **kwargs):
        if self._verbose is False:
            return
        caller = getframeinfo(stack()[1][0])
        print("{time} {file}:{line} - {s}".format(time=datetime.now(pytz.timezone('Israel')).strftime("%d-%m-%Y %H:%M:%S"),
                                                  file=os.path.basename(caller.filename),
                                                  line=caller.lineno,
                                                  s=args[0]), *args[1:], **kwargs)

