import torch
import psutil
from contextlib import contextmanager
import time
from time import perf_counter


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, ' \
           f'GPU memory reserved: {torch.cuda.memory_reserved() /1024**3:.2f} GB, ' \
           f'CPU memory usage: {psutil.Process().memory_info().rss / 1024**3:.2f} GB'


@contextmanager
def compute_throughput(batch_size) -> float:
    start = perf_counter()
    yield lambda: batch_size / ((perf_counter() - start) * 1000)


@contextmanager
def get_time_elapsed(logger, repr: str):
    timer = Timer()
    timer.start()
    yield
    elapsed = timer.stop()
    logger.info(f"Time elapsed for {repr}: {elapsed:.4f}s", ranks=[0])


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


def get_partition(embedding_dim, rank, world_size):
    if world_size == 1:
        return 0, embedding_dim, True

    assert embedding_dim >= world_size, \
        f"Embedding dimension {embedding_dim} must be larger than the world size " \
        f"{world_size} of the process group"
    chunk_size = embedding_dim // world_size
    threshold = embedding_dim % world_size
    # if embedding dim is divisible by world size
    if threshold == 0:
        return rank * chunk_size, (rank + 1) * chunk_size, True

    # align with the split strategy of torch.tensor_split
    size_list = [chunk_size + 1 if i < threshold else chunk_size for i in range(world_size)]
    offset = sum(size_list[:rank])
    return offset, offset + size_list[rank], False
