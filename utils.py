from dataclasses import dataclass, field
from typing import List, Optional
import psutil

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
    output = p.key_averages().table(sort_by="cpu_time_total", row_limit=20)
    if dist.get_rank() == 0:
        print(output)
        # p.export_chrome_trace("tmp/trace_" + str(p.step_num) + ".json")
