import os

import torch
import torch.distributed as dist
import cupy
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch

from recsys.datasets.criteo import get_id_freq_map

INPUT_DATA_DIR = "/data/criteo_preproc/train/"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16384))
PARTS_PER_CHUNK = int(os.environ.get("PARTS_PER_CHUNK", 2))
CONTINUOUS_COLUMNS = ["int_" + str(x) for x in range(0, 13)]
CATEGORICAL_COLUMNS = ["cat_" + str(x) for x in range(0, 26)]
LABEL_COLUMNS = ["label"]


class KJTTransform:

    def __init__(self, dataloader):
        self.batch_size = dataloader.batch_size
        self.cats = dataloader.cat_names
        self.conts = dataloader.cont_names
        self.labels = dataloader.label_names

        _num_ids_in_batch = len(self.cats) * self.batch_size
        self.lengths = torch.ones((_num_ids_in_batch,), dtype=torch.int32)
        self.offsets = torch.arange(0, _num_ids_in_batch + 1, dtype=torch.int32)
        self.length_per_key = len(self.cats) * [self.batch_size]
        self.offset_per_key = [self.batch_size * i for i in range(len(self.cats) + 1)]
        self.index_per_key = {key: i for (i, key) in enumerate(self.cats)}

    def transform(self, batch):
        sparse, dense = [], []
        for col in self.cats:
            sparse.append(batch[0][col])
        sparse = torch.cat(sparse, dim=1)
        for col in self.conts:
            dense.append(batch[0][col])
        dense = torch.cat(dense, dim=1)

        return Batch(
            dense_features=dense,
            sparse_features=KeyedJaggedTensor(
                keys=self.cats,
                values=sparse.transpose(1, 0).reshape(-1),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.batch_size,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=batch[1],
        )


def seed_fn():
    """
    Generate consistent dataloader shuffle seeds across workers
    Reseeds each worker's dataloader each epoch to get fresh a shuffle
    that's consistent across workers.
    """

    max_rand = torch.iinfo(torch.int).max // world_size

    # Generate a seed fragment
    seed_fragment = cupy.random.randint(0, max_rand)

    # Aggregate seed fragments from all workers
    seed_tensor = torch.tensor(seed_fragment)    # pylint: disable=not-callable
    dist.all_reduce(seed_tensor, op=dist.ReduceOp.SUM)
    return seed_tensor % max_rand


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print(f"init rank: {rank}")
    torch.cuda.set_device(rank)

    id_freq_map = get_id_freq_map("/data/criteo_preproc")
    print(
        f"rank: {rank}, shape: {id_freq_map.shape}, max: {id_freq_map.max().item()}, min: {id_freq_map.min().item()}, "
        f"top 10: {id_freq_map[:10].tolist()}")


if __name__ == "__main__":
    # world_size = int(os.environ["OMPI_COMM_WORLD_SIZE"])
    # world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    os.environ["LIBCUDF_CUFILE_POLICY"] = "ALWAYS"
    run(0, 1)
