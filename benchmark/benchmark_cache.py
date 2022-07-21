"""
1. hit rate
2. bandwidth
"""
import itertools
from tqdm import tqdm
from contexttimer import Timer

import torch

from recsys.modules.embeddings import CachedEmbeddingBag, FreqAwareEmbeddingBag
from data_utils import get_dataloader, get_id_freq_map, NUM_EMBED


def main(batch_size, embedding_dim, cache_sets, cache_lines, embed_type, id_freq_map=None):
    dataloader = get_dataloader('train', batch_size)
    print(f"batch size: {batch_size}, num of batches: {len(dataloader)}")
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
        model = FreqAwareEmbeddingBag(NUM_EMBED, embedding_dim, sparse=True, include_last_offset=True).to(device)
        model._preprocess(cache_lines, cache_sets, id_freq_map)
    else:
        raise RuntimeError(f"Unknown EB type: {embed_type}")

    chunk_size_in_bytes = embedding_dim * cache_lines * model.weight.element_size()
    grad = None
    with tqdm(bar_format='{n_fmt}it {rate_fmt} {postfix}') as t:
        for it in itertools.count():
            batch = next(data_iter)
            sparse_feature = batch.sparse_features.to(device)

            res = model(sparse_feature.values(), sparse_feature.offsets())

            # grad = torch.randn_like(res) if grad is None else grad
            # res.backward(grad)
            #
            # model.zero_grad()
            hit_rate = model.num_hits_history[it] / (model.num_hits_history[it] + model.num_miss_history[it])
            t.set_postfix_str(f"hit_rate={hit_rate*100:.2f}%, "
                              f"swap in bandwidth={model.swap_in_bandwidth:.2f} MB/s, "
                              f"swap out bandwidth={model.swap_out_bandwidth:.2f} MB/s")
            t.update()
            if it == 15:
                break


if __name__ == "__main__":
    with Timer() as timer:
        id_freq_map = get_id_freq_map()
    print(f"Counting sparse features in dataset costs: {timer.elapsed:.2f} s")

    batch_size = [2048, 4096, 8192, 16384]
    embed_dim = 128
    cache_sets = [50_000, 500_000]
    cache_lines = [64, 128, 256, 512]

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
