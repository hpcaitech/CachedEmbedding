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

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from colossalai.nn.parallel.layers import FreqAwareEmbeddingBag, EvictionStrategy
from recsys.datasets.criteo import get_id_freq_map
from data_utils import get_dataloader, NUM_EMBED, CRITEO_PATH


def benchmark_cache_embedding(batch_size,
                              embedding_dim,
                              cache_ratio,
                              id_freq_map=None,
                              warmup_ratio=0.,
                              use_limit_buf=True,
                              use_lfu=False):
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
    device = torch.device('cuda:0')
    with Timer() as timer:
        model = FreqAwareEmbeddingBag(NUM_EMBED, embedding_dim, sparse=True, include_last_offset=True, evict_strategy=EvictionStrategy.LFU if use_lfu else EvictionStrategy.DATASET).to(device)
        print(f"model init: {timer.elapsed:.2f}s")
        with Timer() as timer:
            model.preprocess(cuda_row_num, id_freq_map, warmup_ratio=warmup_ratio, buffer_size=buf_size)
        print(f"reorder: {timer.elapsed:.2f}s")

    grad = None
    avg_hit_rate = None
    print(
        f'after reorder max_memory_allocated {torch.cuda.max_memory_allocated()/1e9} GB, max_memory_reserved {torch.cuda.max_memory_allocated()/1e9} GB'
    )
    torch.cuda.reset_peak_memory_stats()

    with Timer() as timer:
        with tqdm(bar_format='{n_fmt}it {rate_fmt} {postfix}') as t:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            #              schedule=schedule(wait=0, warmup=21, active=2, repeat=1),
            #              profile_memory=True,
            #              on_trace_ready=tensorboard_trace_handler(
            #                  f"log/b{batch_size}-e{embedding_dim}-num_chunk{cuda_row_num}-chunk_size{cache_lines}")) as prof:
            with nullcontext():
                for it in itertools.count():
                    batch = next(data_iter)
                    sparse_feature = batch.sparse_features.to(device)

                    res = model(sparse_feature.values(), sparse_feature.offsets())

                    grad = torch.randn_like(res) if grad is None else grad
                    res.backward(grad)

                    model.zero_grad()
                    # prof.step()
                    running_hits = model.num_hits_history[-1]    # sum(model.num_hits_history)
                    running_miss = model.num_miss_history[-1]    # sum(model.num_miss_history)
                    hit_rate = running_hits / (running_hits + running_miss)
                    t.set_postfix_str(f"hit_rate={hit_rate*100:.2f}%, "
                                      f"swap in bandwidth={model.swap_in_bandwidth:.2f} MB/s, "
                                      f"swap out bandwidth={model.swap_out_bandwidth:.2f} MB/s")
                    t.update()
                    if it == 200:
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