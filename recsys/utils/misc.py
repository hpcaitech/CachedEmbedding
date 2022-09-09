import torch
import psutil
from contextlib import contextmanager
import time
from time import perf_counter
from dataclasses import dataclass, field
from typing import List, Optional
from colossalai.nn.parallel.layers import TablewiseEmbeddingBagConfig

import numpy as np

@dataclass
class TrainValTestResults:
    val_accuracies: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)
    test_accuracy: Optional[float] = None
    test_auroc: Optional[float] = None

def count_parameters(model, prefix=''):
    trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_amount = sum(p.numel() for p in model.parameters())
    buffer_amount = sum(b.numel() for b in model.buffers())
    param_storage = sum([p.numel() * p.element_size() for p in model.parameters()])
    buffer_storage = sum([b.numel() * b.element_size() for b in model.buffers()])
    stats_str = f'{prefix}: {trainable_param:,}.' + '\n'
    stats_str += f"Number of model parameters: {param_amount:,}, storage overhead: {param_storage/1024**3:.2f} GB. "
    stats_str += f"Number of model buffers: {buffer_amount:,}, storage overhead: {buffer_storage/1024**3:.2f} GB."
    return stats_str


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
        torch.cuda.synchronize()
        return time.time()

    def start(self):
        """Firstly synchronize cuda, reset the clock and then start the timer.
        """
        self._elapsed = 0
        torch.cuda.synchronize()
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
        torch.cuda.synchronize()
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


def prepare_tablewise_config(num_embeddings_per_feature,
                             cache_ratio,
                             id_freq_map_total=None,
                             dataset="criteo_kaggle",
                             world_size=2):
    # WARNING, prototype. only support criteo_kaggle dataset and world_size == 2, 4
    # TODO: automatic arrange
    embedding_bag_config_list: List[TablewiseEmbeddingBagConfig] = []
    rank_arrange = get_tablewise_rank_arrange(dataset, world_size)
    table_offsets = np.array([0, *np.cumsum(num_embeddings_per_feature)])
    for i, num_embeddings in enumerate(num_embeddings_per_feature):
        ids_freq_mapping = None
        if id_freq_map_total != None:
            ids_freq_mapping = id_freq_map_total[table_offsets[i] : table_offsets[i + 1]]
        cuda_row_num = int(cache_ratio * num_embeddings) + 2000
        if cuda_row_num > num_embeddings:
            cuda_row_num = num_embeddings
        embedding_bag_config_list.append(
            TablewiseEmbeddingBagConfig(
                num_embeddings=num_embeddings,
                cuda_row_num=cuda_row_num,
                assigned_rank=rank_arrange[i],
                ids_freq_mapping=ids_freq_mapping
            )
        )
    return embedding_bag_config_list

def get_tablewise_rank_arrange(dataset=None, world_size=0):
    if 'criteo' in dataset and 'kaggle' in dataset:
        if world_size == 1:
            rank_arrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif world_size == 2:
            rank_arrange = [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0]
        elif world_size == 3:
            rank_arrange = [2, 1, 0, 1, 1, 2, 2, 1, 0, 0, 1, 1, 0, 1, 0, 2, 0, 2, 2, 0, 2, 2, 0, 1, 1, 0]
        elif world_size == 4:
            rank_arrange = [3, 1, 0, 3, 1, 0, 2, 1, 0, 2, 3, 1, 3, 1, 2, 3, 1, 2, 3, 0, 2, 0, 0, 2, 3, 2]
        elif world_size == 8:
            rank_arrange = [6, 6, 0, 4, 7, 2, 5, 7, 0, 5, 7, 1, 7, 3, 5, 3, 1, 6, 6, 0, 2, 2, 1, 4, 3, 4]
        else :
            raise NotImplementedError("Other Tablewise settings are under development")
    elif 'criteo' in dataset:
        if world_size == 1:
            rank_arrange = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        elif world_size == 2:
            rank_arrange = [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
        elif world_size == 4:
            rank_arrange = [1, 3, 3, 3, 3, 0, 2, 2, 1, 2, 2, 2, 0, 1, 2, 1, 0, 1, 0, 0, 2, 3, 3, 3, 1, 0]
        else :
            raise NotImplementedError("Other Tablewise settings are under development")
    else:
        raise NotImplementedError("Other Tablewise settings are under development")
    return rank_arrange