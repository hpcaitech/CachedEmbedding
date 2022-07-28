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

from recsys.modules.embeddings import CachedEmbeddingBag, FreqAwareEmbeddingBag
from data_utils import get_dataloader, get_id_freq_map, NUM_EMBED


def main(batch_size, embedding_dim, cache_sets, cache_lines, embed_type, id_freq_map=None):
    dataloader = get_dataloader('train', batch_size)
    print(f"batch size: {batch_size}, num of batches: {len(dataloader)}, "
          f"cached chunks: {cache_sets}, chunk size: {cache_lines}")
    data_iter = iter(dataloader)

    device = torch.device('cuda:0')
    if embed_type == 'row':
        model = CachedEmbeddingBag(NUM_EMBED,
                                   embedding_dim,
                                   cache_sets,
                                   cache_lines=cache_lines,
                                   sparse=True,
                                   include_last_offset=True).to(device)
    elif embed_type == 'chunk':
        with Timer() as timer:
            model = FreqAwareEmbeddingBag(NUM_EMBED, embedding_dim, sparse=True, include_last_offset=True).to(device)
        print(f"model init: {timer.elapsed:.2f}s")
        with Timer() as timer:
            model._preprocess(cache_lines, cache_sets, id_freq_map)
        print(f"reorder: {timer.elapsed:.2f}s")
    else:
        raise RuntimeError(f"Unknown EB type: {embed_type}")

    # grad = None
    hist_str = None
    with tqdm(bar_format='{n_fmt}it {rate_fmt} {postfix}') as t:
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              schedule=schedule(wait=0, warmup=21, active=2, repeat=1),
        #              profile_memory=True,
        #              on_trace_ready=tensorboard_trace_handler(
        #                  f"log/b{batch_size}-e{embedding_dim}-num_chunk{cache_sets}-chunk_size{cache_lines}")) as prof:
        with nullcontext():
            for it in itertools.count():
                batch = next(data_iter)
                sparse_feature = batch.sparse_features.to(device)

                with torch.no_grad():
                    res = model.chunk_weight_mgr.prepare_ids(sparse_feature.values())

                # grad = torch.randn_like(res) if grad is None else grad
                # res.backward(grad)
                #
                # model.zero_grad()
                # prof.step()
                running_hits = model.num_hits_history[-1]    # sum(model.num_hits_history)
                running_miss = model.num_miss_history[-1]    # sum(model.num_miss_history)
                hit_rate = running_hits / (running_hits + running_miss)
                t.set_postfix_str(f"hit_rate={hit_rate*100:.2f}%, "
                                  f"swap in bandwidth={model.swap_in_bandwidth:.2f} MB/s, "
                                  f"swap out bandwidth={model.swap_out_bandwidth:.2f} MB/s")
                t.update()
                if it == 100:
                    hit_hist = np.array(model.num_hits_history)
                    miss_hist = np.array(model.num_miss_history)
                    hist = hit_hist / (hit_hist + miss_hist)
                    hist_str = '\n'.join([f"{it}it: {_h*100:.2f} %" for it, _h in enumerate(hist.tolist())])
                    break
    print(f"hit rate history: {hist_str}")


if __name__ == "__main__":
    with Timer() as timer:
        id_freq_map = get_id_freq_map()
    print(f"Counting sparse features in dataset costs: {timer.elapsed:.2f} s")

    batch_size = [2048]
    embed_dim = 32
    cache_sets = [20_000, 50_000]
    # chunk size
    cache_lines = [512]

    # # row-wise cache
    # for bs in batch_size:
    #     for cs in cache_sets:
    #         main(bs, embed_dim, cache_sets=cs, cache_lines=1, embed_type='row')

    # chunk-wise cache
    for bs in batch_size:
        for cs in cache_sets:
            for cl in cache_lines:
                try:
                    main(bs, embed_dim, cache_sets=cs, cache_lines=cl, embed_type='chunk', id_freq_map=id_freq_map)
                    print('=' * 50 + '\n')

                except AssertionError as ae:
                    print(f"batch size: {bs}, num cache sets: {cs}, num cache lines: {cl}, raise error: {ae}")
                    print('=' * 50 + '\n')
