import logging
from typing import List, Optional
import inspect

from .distributed_manager import ParallelMode, DISTMGR as dist_manager

_SYS = 'recsys'
_FORMAT = '%(name)s - %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=_FORMAT)


def disable_existing_loggers(include: Optional[List[str]] = None, exclude: List[str] = [_SYS]) -> None:
    """Set the level of existing loggers to `WARNING`.
    By default, it will "disable" all existing loggers except the logger named "recsys".

    Args:
        include (Optional[List[str]], optional): Loggers whose name in this list will be disabled.
            If set to `None`, `exclude` argument will be used. Defaults to None.
        exclude (List[str], optional): Loggers whose name not in this list will be disabled.
            This argument will be used only when `include` is None. Defaults to ['recsys'].
    """
    if include is None:
        filter_func = lambda name: name not in exclude
    else:
        filter_func = lambda name: name in include

    for log_name in logging.Logger.manager.loggerDict.keys():
        if filter_func(log_name):
            logging.getLogger(log_name).setLevel(logging.WARNING)


def get_distributed_logger(name=_SYS):
    return DistributedLogger.get_instance(name)


class DistributedLogger:
    """This is a distributed event logger class essentially based on :class:`logging`.

    Args:
        name (str): The name of the logger.

    Note:
        The parallel_mode used in ``info``, ``warning``, ``debug`` and ``error``
        should be concluded in ``ParallelMode``. More details about ``ParallelMode`` could be found
        in `parallel_mode <https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/context/parallel_mode.py>`_.
    """

    __instances = dict()

    @staticmethod
    def get_instance(name: str):
        """Get the unique single logger instance based on name.

        Args:
            name (str): The name of the logger.

        Returns:
            DistributedLogger: A DistributedLogger object
        """
        if name in DistributedLogger.__instances:
            return DistributedLogger.__instances[name]
        else:
            logger = DistributedLogger(name=name)
            return logger

    def __init__(self, name):
        if name in DistributedLogger.__instances:
            raise Exception(
                'Logger with the same name has been created, you should use colossalai.logging.get_dist_logger')
        else:
            self._name = name
            self._logger = logging.getLogger(name)
            DistributedLogger.__instances[name] = self

    @staticmethod
    def __get_call_info():
        stack = inspect.stack()

        # stack[1] gives previous function ('info' in our case)
        # stack[2] gives before previous function and so on

        fn = stack[2][1]
        ln = stack[2][2]
        func = stack[2][3]

        return fn, ln, func

    @staticmethod
    def _check_valid_logging_level(level: str):
        assert level in ['INFO', 'DEBUG', 'WARNING', 'ERROR'], 'found invalid logging level'

    def set_level(self, level: str) -> None:
        """Set the logging level

        Args:
            level (str): Can only be INFO, DEBUG, WARNING and ERROR.
        """
        self._check_valid_logging_level(level)
        self._logger.setLevel(getattr(logging, level))

    def _log(self,
             level,
             message: str,
             parallel_mode: ParallelMode = ParallelMode.DEFAULT,
             ranks: List[int] = None) -> None:
        if ranks is None:
            getattr(self._logger, level)(message)
        else:
            local_rank = dist_manager.get_rank(parallel_mode)
            if local_rank in ranks:
                getattr(self._logger, level)(message)

    def info(self, message: str, parallel_mode: ParallelMode = ParallelMode.DEFAULT, ranks: List[int] = None) -> None:
        """Log an info message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        # message_prefix = "{}:{} {}".format(*self.__get_call_info())
        # self._log('info', message_prefix, parallel_mode, ranks)
        self._log('info', message, parallel_mode, ranks)

    def warning(self,
                message: str,
                parallel_mode: ParallelMode = ParallelMode.DEFAULT,
                ranks: List[int] = None) -> None:
        """Log a warning message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('warning', message_prefix, parallel_mode, ranks)
        self._log('warning', message, parallel_mode, ranks)

    def debug(self, message: str, parallel_mode: ParallelMode = ParallelMode.DEFAULT, ranks: List[int] = None) -> None:
        """Log a debug message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('debug', message_prefix, parallel_mode, ranks)
        self._log('debug', message, parallel_mode, ranks)

    def error(self, message: str, parallel_mode: ParallelMode = ParallelMode.DEFAULT, ranks: List[int] = None) -> None:
        """Log an error message.

        Args:
            message (str): The message to be logged.
            parallel_mode (:class:`colossalai.context.parallel_mode.ParallelMode`):
                The parallel mode used for logging. Defaults to ParallelMode.GLOBAL.
            ranks (List[int]): List of parallel ranks.
        """
        message_prefix = "{}:{} {}".format(*self.__get_call_info())
        self._log('error', message_prefix, parallel_mode, ranks)
        self._log('error', message, parallel_mode, ranks)


distributed_logger = get_distributed_logger()
