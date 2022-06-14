import torch
import psutil
from contextlib import contextmanager
import time


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, ' \
           f'CPU memory usage: {psutil.Process().memory_info().rss / 1024**3:.2f} GB'


@contextmanager
def get_time_elapsed(logger, repr: str):
    timer = Timer()
    timer.start()
    yield
    elapsed = timer.stop()
    logger.info(f"Time elapsed for {repr}: {elapsed:.4f}s", ranks=[0])


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class Timer:
    """A timer object which helps to log the execution times, and provides different tools to assess the times.
    """

    def __init__(self):
        self._started = False
        self._start_time = time.time()
        self._elapsed = 0
        self._history = []

    @property
    def has_history(self):
        return len(self._history) != 0

    @property
    def current_time(self) -> float:
        synchronize()
        return time.time()

    def start(self):
        """Firstly synchronize cuda, reset the clock and then start the timer.
        """
        self._elapsed = 0
        synchronize()
        self._start_time = time.time()
        self._started = True

    def lap(self):
        """lap time and return elapsed time
        """
        return self.current_time - self._start_time

    def stop(self, keep_in_history: bool = False):
        """Stop the timer and record the start-stop time interval.

        Args:
            keep_in_history (bool, optional): Whether does it record into history
                each start-stop interval, defaults to False.
        Returns:
            int: Start-stop interval.
        """
        synchronize()
        end_time = time.time()
        elapsed = end_time - self._start_time
        if keep_in_history:
            self._history.append(elapsed)
        self._elapsed = elapsed
        self._started = False
        return elapsed

    def get_history_mean(self):
        """Mean of all history start-stop time intervals.

        Returns:
            int: Mean of time intervals
        """
        return sum(self._history) / len(self._history)

    def get_history_sum(self):
        """Add up all the start-stop time intervals.

        Returns:
            int: Sum of time intervals.
        """
        return sum(self._history)

    def get_elapsed_time(self):
        """Return the last start-stop time interval.

        Returns:
            int: The last time interval.

        Note:
            Use it only when timer is not in progress
        """
        assert not self._started, 'Timer is still in progress'
        return self._elapsed

    def reset(self):
        """Clear up the timer and its history
        """
        self._history = []
        self._started = False
        self._elapsed = 0
