import logging
from datetime import datetime
from pytz import timezone, utc


LOG_LEVEL = logging.DEBUG


class Verbose(object):
    """ Inherit this class to call self._print and get line number etc."""
    def __init__(self, name=None):
        self._name = name if name else self.__class__.__name__
        self._init_logging()

    def _init_logging(self):
        logging.basicConfig(level=LOG_LEVEL,
                            format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
        logging.Formatter.converter = Verbose.custom_time
        self.logger = logging.getLogger(self._name)

    @staticmethod
    def custom_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("Israel")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
