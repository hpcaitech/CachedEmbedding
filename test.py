import os
import time
import os
import shutil

from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from nvtabular.utils import device_mem_size
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import nvtabular as nvt
from nvtabular.loader.torch import TorchAsyncItr    # , DLDataLoader
import cupy
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch

import colossalai
from recsys.datasets.criteo import get_id_freq_map
from recsys.utils import get_mem_info

INPUT_DATA_DIR = "/data/criteo_preproc/test/"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 16384))
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


def setup_dask(dask_workdir):
    if os.path.exists(dask_workdir):
        shutil.rmtree(dask_workdir)
    os.makedirs(dask_workdir)

    device_limit_frac = 0.05    # Spill GPU-Worker memory to host at this limit.
    device_pool_frac = 0.04

    # Use total device size to calculate device limit and pool_size
    device_size = device_mem_size(kind="total")
    device_limit = int(device_limit_frac * device_size)
    device_pool_size = int(device_pool_frac * device_size)

    cluster = LocalCUDACluster(
        protocol="tcp",
        n_workers=1,
        CUDA_VISIBLE_DEVICES=os.environ["CUDA_VISIBLE_DEVICES"],
        device_memory_limit=device_limit,
        local_directory=dask_workdir,
        rmm_pool_size=(device_pool_size // 256) * 256,
    )

    return Client(cluster)


def run():
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    os.environ["LOCAL_RANK"] = '0'

    colossalai.logging.disable_existing_loggers()
    colossalai.launch_from_torch(config={}, verbose=False)

    fname = "part_{}.parquet"
    train_paths = [os.path.join(INPUT_DATA_DIR, fname.format(i)) for i in range(64)]

    print(f"{dist.get_rank()}/{dist.get_world_size()}: device: {torch.cuda.current_device()}")

    start = time.time()
    train_data = nvt.Dataset(train_paths, engine="parquet", part_mem_fraction=0.02)
    print(f"nvdtaset: {time.time() - start}")
    start = time.time()
    train_data_idrs = TorchAsyncItr(
        train_data,
        batch_size=BATCH_SIZE,
        cats=CATEGORICAL_COLUMNS,
        conts=CONTINUOUS_COLUMNS,
        labels=LABEL_COLUMNS,
        global_rank=0,
        global_size=1,
        drop_last=True,
        shuffle=True,
        seed_fn=lambda: 1,
    )
    print(f"TorchAsyncItr: {time.time() - start}, len: {len(train_data_idrs)}")

    start = time.time()
    train_dataloader = DataLoader(train_data_idrs,
                                  collate_fn=KJTTransform(train_data_idrs).transform,
                                  batch_size=None,
                                  pin_memory=False,
                                  num_workers=0)
    print(f"dataloader: {time.time() - start}, len: {len(train_dataloader)}")

    data_iter = iter(train_dataloader)
    for idx, batch in enumerate(data_iter):
        print(f"rank: {dist.get_rank()}, it: {idx}, batch: {batch.dense_features}")

        if idx == 30:
            break
    print(get_mem_info())
    torch.cuda.synchronize()
    # id_freq_map = get_id_freq_map("/data/criteo_preproc")
    # print(id_freq_map.shape, id_freq_map.max(), id_freq_map.min())


if __name__ == "__main__":
    os.environ["LIBCUDF_CUFILE_POLICY"] = "ALWAYS"

    run()
