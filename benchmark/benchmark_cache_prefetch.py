"""
1. hit rate
2. bandwidth
3. read / load
4. elapsed time
"""
import itertools
from tqdm import tqdm
from contexttimer import Timer
from contextlib import nullcontext
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function

from colossalai.nn.parallel.layers import FreqAwareEmbeddingBag
from recsys.datasets.criteo import get_id_freq_map
from data_utils import get_dataloader, NUM_EMBED, CRITEO_PATH


# custom pipeline for freqaware embedding
def _to_device(batch, device: torch.device, non_blocking: bool):
    return batch.to(device=device, non_blocking=non_blocking)

def _wait_for_batch(batch, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    cur_stream = torch.cuda.current_stream()
    batch.record_stream(cur_stream)

class TrainPipelineBase:
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
    ) -> None:
        self._model = model
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device is not None and device.type == "cuda" else None
        )
        self._cur_batch = None
        self._connected = False

    def _connect(self, dataloader_iter) -> None:
        cur_batch = next(dataloader_iter).sparse_features
        self._cur_batch = cur_batch
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
        self._connected = True

    def progress(self, dataloader_iter):
        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch = next(dataloader_iter).sparse_features
        
        cur_batch = self._cur_batch
        assert cur_batch is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._model.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)

        with record_function("## forward ##"):
            output = self._model(cur_batch.values(), cur_batch.offsets())
            
        with record_function("## backward ##"):
            grad = torch.randn_like(output)
            output.backward(grad)

        # Copy the next batch to GPU
        self._cur_batch = next_batch

        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(self._cur_batch, self._device, non_blocking=True)
        
        # running_hits = self._model.num_hits_history[-1]    # sum(model.num_hits_history)
        # running_miss = self._model.num_miss_history[-1]    # sum(model.num_miss_history)
        # hit_rate = running_hits / (running_hits + running_miss)
        # print(f"hit_rate={hit_rate*100:.2f}%, "
        #                               f"swap in bandwidth={self._model.swap_in_bandwidth:.2f} MB/s, "
        #                               f"swap out bandwidth={self._model.swap_out_bandwidth:.2f} MB/s")


def benchmark_cache_embedding(batch_size,
                              embedding_dim,
                              cache_ratio,
                              id_freq_map=None,
                              warmup_ratio=0.,
                              use_limit_buf=True):
    dataloader = get_dataloader('train', batch_size)
    cuda_row_num = int(cache_ratio * NUM_EMBED)
    print(f"batch size: {batch_size}, "
          f"num of batches: {len(dataloader)}, "
          f"cached rows: {cuda_row_num},  cached_ratio {cuda_row_num / NUM_EMBED}")
    data_iter = iter(dataloader)

    buf_size = 0
    if use_limit_buf:
        buf_size = int(np.ceil(cuda_row_num * 0.1))

    torch.cuda.reset_peak_memory_stats()
    device = torch.device('cuda')
    with Timer() as timer:
            model = FreqAwareEmbeddingBag(NUM_EMBED, embedding_dim, sparse=True, include_last_offset=True,
                                        cuda_row_num=cuda_row_num,
                                        ids_freq_mapping=id_freq_map,
                                        warmup_ratio=warmup_ratio,
                                        buffer_size=buf_size, ).to(device)
            print(f"model init: {timer.elapsed:.2f}s")

    avg_hit_rate = None
    print(
        f'after reorder max_memory_allocated {torch.cuda.max_memory_allocated()/1e9} GB, max_memory_reserved {torch.cuda.max_memory_allocated()/1e9} GB'
    )
    torch.cuda.reset_peak_memory_stats()
    
    pipe = TrainPipelineBase(model, device)

    with Timer() as timer:
        # with nullcontext():
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                        schedule=schedule(wait=0, warmup=21, active=2, repeat=1),
                        profile_memory=True,
                        on_trace_ready=tensorboard_trace_handler(
                            f"log/prefetch-b{batch_size}-e{embedding_dim}-num_chunk{cuda_row_num}")) as prof:
            count = 0
            for it in tqdm(itertools.count()):
                try:
                    pipe.progress(data_iter)
                    prof.step()
                    count += 1
                    if count == 200:
                        break
                except StopIteration:
                    break
            
    hit_hist = np.array(model.num_hits_history)
    miss_hist = np.array(model.num_miss_history)
    hist = hit_hist / (hit_hist + miss_hist)
    avg_hit_rate = np.mean(hist)
    print(f"average hit rate: {avg_hit_rate}")
    model.cache_weight_mgr.print_comm_stats()
    print(
        f'training max_memory_allocated {torch.cuda.max_memory_allocated()/1e9} GB, max_memory_reserved {torch.cuda.max_memory_allocated()/1e9} GB'
    )
    print(f'overall training time {timer.elapsed:.2f}s')


if __name__ == "__main__":
    with Timer() as timer:
        id_freq_map = get_id_freq_map(CRITEO_PATH)
    print(f"Counting sparse features in dataset costs: {timer.elapsed:.2f} s")

    batch_size = [2048]
    embed_dim = 32
    cache_ratio = [0.02]

    # # row-wise cache
    # for bs in batch_size:
    #     for cs in cuda_row_num:
    #         main(bs, embed_dim, cuda_row_num=cs, cache_lines=1, embed_type='row')

    # chunk-wise cache
    for bs in batch_size:
        for cr in cache_ratio:
            for warmup_ratio in [0.7]:
                for use_buf in [False, True]:
                    try:
                        benchmark_cache_embedding(bs,
                                                    embed_dim,
                                                    cache_ratio=cr,
                                                    id_freq_map=id_freq_map,
                                                    warmup_ratio=warmup_ratio,
                                                    use_limit_buf=use_buf)
                        print('=' * 50 + '\n')

                    except AssertionError as ae:
                        print(f"batch size: {bs}, cache ratio: {cr}, raise error: {ae}")
                        print('=' * 50 + '\n')
