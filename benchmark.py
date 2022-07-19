"""
1. hit rate
2. bandwidth
"""
import os
import itertools
from tqdm import tqdm
import time

import torch
from torch.utils.data import DataLoader
from recsys.datasets import criteo
from recsys.modules.embeddings import CachedEmbeddingBag

CRITEO_PATH = "criteo_kaggle"
NUM_EMBED = 33762577


def get_dataloader(stage, batch_size):
    hash_sizes = list(map(int, criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(',')))
    files = os.listdir(CRITEO_PATH)

    def is_final_day(s):
        return "day_6" in s

    rank, world_size = 0, 1
    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = rank if stage == "val" else (rank + world_size)
        world_size = world_size * 2

    stage_files = [
        sorted(map(
            lambda x: os.path.join(CRITEO_PATH, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        criteo.InMemoryBinaryCriteoIterDataPipe(
            *stage_files,    # pyre-ignore[6]
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=True,
            hashes=hash_sizes),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    return dataloader


def main(batch_size, embedding_dim, cache_sets):
    dataloader = get_dataloader('train', batch_size)
    print(f"num of batches: {len(dataloader)}")
    data_iter = iter(dataloader)

    device = torch.device('cuda:0')
    model = CachedEmbeddingBag(NUM_EMBED, embedding_dim, cache_sets, sparse=True, include_last_offset=True).to(device)

    grad = None
    with tqdm(bar_format='{n_fmt}it {rate_fmt} {postfix}') as t:
        for it in itertools.count():
            batch = next(data_iter)
            sparse_feature = batch.sparse_features.to(device)

            start = time.time()
            res = model(sparse_feature.values(), sparse_feature.offsets())
            torch.cuda.synchronize()
            duration = time.time() - start
            grad = torch.randn_like(res) if grad is None else grad
            res.backward(grad)

            model.zero_grad()
            hit_rate = model.num_hits_history[it] / (model.num_hits_history[it] + model.num_miss_history[it])
            bandwidth = model.num_miss_history[it] * embedding_dim * 4 / 1024**2
            t.set_postfix_str(f"hit_rate={hit_rate:.2f}, bandwidth={bandwidth/duration:.2f}MB/s")
            t.update()
            if it == 50:
                break


if __name__ == "__main__":
    main(2048, 128, 500_000)
