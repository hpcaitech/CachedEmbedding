import os
from tqdm import tqdm
import itertools
import random

import torch
import torch.distributed as dist
from recsys.datasets.criteo import PetastormDataReader, get_id_freq_map
from recsys.utils import get_mem_info
from baselines.data.dlrm_dataloader import PetastormDataReader as Reader
import pyarrow.parquet as pq


def iterate_data():
    # dist.init_process_group(backend='nccl')

    dataset_dir = "/data/criteo_preproc/validation/"
    fname = "part_{}.parquet"
    train_paths = [dataset_dir + fname.format(i) for i in range(64)]

    reader = Reader(train_paths, batch_size=16384)
    for batch in reader:
        print(batch)
        break

    # random.seed(0)
    # dataloader = PetastormDataReader(train_paths, batch_size=16384, rank=None, world_size=None, shuffle_batches=False)
    #
    # data_iter = iter(dataloader)
    # for idx in tqdm(itertools.count(), ncols=0, total=len(dataloader) if hasattr(dataloader, "__len__") else None):
    #     batch = next(data_iter)
    #     # print(f"rank: {dist.get_rank()}, it {idx}, dense: {batch.dense_features[:5, :5]}")
    #     # if idx == 2:
    #     #     break

    # dataset_dir = "/data/criteo_preproc/"
    # id_freq_map = get_id_freq_map(dataset_dir)
    #
    # print(f"rank: {dist.get_rank()}, first 10: {id_freq_map[:10]}")


if __name__ == "__main__":
    iterate_data()
