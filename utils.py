from dataclasses import dataclass, field
from typing import List, Optional
import psutil
from contextlib import contextmanager
from colossalai.utils import Timer
from colossalai.logging import get_dist_logger
import torch
import torch.distributed as dist


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB, ' \
           f'CPU memory usage: {psutil.Process().memory_info().rss / 1024**3:.2f} GB'


@dataclass
class TrainValTestResults:
    val_accuracies: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)
    test_accuracy: Optional[float] = None
    test_auroc: Optional[float] = None


def trace_handler(p):
    stats = p.key_averages()
    if dist.get_rank() == 0:
        stats_str = "CPU Time Total:\n" + f"{stats.table(sort_by='cpu_time_total', row_limit=20)}" + "\n"
        stats_str += "Self CPU Time Total:\n" + f"{stats.table(sort_by='self_cpu_time_total', row_limit=20)}" + "\n"
        stats_str += "CUDA Time Total:\n" + f"{stats.table(sort_by='cuda_time_total', row_limit=20)}" + "\n"
        stats_str += "Self CUDA Time Total:\n" + f"{stats.table(sort_by='self_cuda_time_total', row_limit=20)}" + "\n"

        print(stats_str)
        # p.export_chrome_trace("tmp/trace_" + str(p.step_num) + ".json")


@contextmanager
def get_time_elapsed(logger, repr: str):
    timer = Timer()
    timer.start()
    yield
    elapsed = timer.stop()
    logger.info(f"Time elapsed for {repr}: {elapsed:.4f}s", ranks=[0])
