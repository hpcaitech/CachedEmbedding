import pytest
from functools import partial
import numpy as np
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import torchrec
from torchrec.datasets.criteo import InMemoryBinaryCriteoIterDataPipe as TorchRecCriteoLoader
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from recsys import launch, disable_existing_loggers
from recsys import DISTMGR
from colossalai.utils import free_port
from recsys.datasets.criteo import KAGGLE_NUM_EMBEDDINGS_PER_FEATURE, \
    InMemoryBinaryCriteoIterDataPipe as RecSysCriteoLoader
from recsys.datasets.utils import KJTAllToAll

BATCH_SIZE = 512
NUM_EMBEDDINGS_PER_FEATURE = list(map(int, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(',')))
DATASET_DIR = '../criteo_kaggle_data'


def test_keyedjaggedtensor():
    device = torch.device('cuda')
    num_embeddings_per_feature = [1000, 987, 1874]
    feature_offsets = torch.from_numpy(np.array([0, *np.cumsum(num_embeddings_per_feature)[:-1]])).to(device)

    # synthesize a batch of 2 samples:
    #
    #     feature 1    |    feature 2    | feature 3
    # [id1, id2, id3]  |   [id1, ]       | [id1, id2]
    # []               |   [id1, id2]    | [id1, ]

    lengths = [3, 0, 1, 2, 2, 1]
    local_inputs = []
    target = []
    batch_size = len(lengths) // len(num_embeddings_per_feature)
    for idx, l in enumerate(lengths):
        high = num_embeddings_per_feature[idx // batch_size]
        offset = feature_offsets[idx // batch_size]

        _local = torch.randint(low=0, high=high, size=(l,), dtype=torch.long, device=device)
        local_inputs.append(_local)
        target.append(_local + offset)

    inputs = torchrec.KeyedJaggedTensor(
        keys=["t_1", "t_2", "t_3"],
        values=torch.cat(local_inputs),
        lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )
    print(inputs)
    keys = inputs.keys()
    assert len(keys) == len(feature_offsets)

    feat_dict = inputs.to_dict()
    flattened = torch.cat([feat_dict[key].values() + offset for key, offset in zip(keys, feature_offsets)])

    target = torch.cat(target)
    assert torch.allclose(flattened, target)


def is_final_day(s: str, total_days=7) -> bool:
    return f"day_{total_days - 1}" in s


def _test_dataloader(stage_files):
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()

    offsets = torch.from_numpy(np.array([0, *np.cumsum(NUM_EMBEDDINGS_PER_FEATURE)[:-1]]).reshape(-1, 1)).int()

    dataloader = DataLoader(
        RecSysCriteoLoader(
            *stage_files,    # pyre-ignore[6]
            batch_size=BATCH_SIZE,
            rank=rank,
            world_size=world_size,
            shuffle_batches=False,
            hashes=NUM_EMBEDDINGS_PER_FEATURE),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    ref_dataloader = DataLoader(
        TorchRecCriteoLoader(
            *stage_files,    # pyre-ignore[6]
            batch_size=BATCH_SIZE,
            rank=rank,
            world_size=world_size,
            shuffle_batches=False,
            hashes=NUM_EMBEDDINGS_PER_FEATURE),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    for cnt, (batch, ref_batch) in enumerate(zip(dataloader, ref_dataloader)):
        dense, sparse, labels = batch.dense_features, batch.sparse_features, batch.labels
        sparse_values = (sparse.values().view(len(sparse.keys()), -1) - offsets).view(-1)
        ref_dense, ref_sparse, ref_labels = ref_batch.dense_features, ref_batch.sparse_features, ref_batch.labels
        assert torch.allclose(dense, ref_dense) and torch.allclose(labels, ref_labels)
        assert torch.allclose(sparse_values, ref_sparse.values().long()) and \
               torch.allclose(sparse.offsets(), ref_sparse.offsets())
        if cnt == 3:
            break


def _test_dist_dataloader(stage_files):
    """
    It seems that there isn't any trivial way to check the correctness of KJTAllToAll,
    as the splitting strategy for raw data is quite different from distributed dataloaders.

    Here, I validate the logic by comparing each #world_size batches generated by RecSysCriteoLoader
    against a whole batch yielded by TorchRecCriteoLoader
    """
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    batch_size = BATCH_SIZE // world_size
    offsets = torch.from_numpy(np.array([0, *np.cumsum(NUM_EMBEDDINGS_PER_FEATURE)[:-1]]).reshape(-1, 1)).int()

    dataloader = DataLoader(
        RecSysCriteoLoader(
            *stage_files,    # pyre-ignore[6]
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=False,
            hashes=NUM_EMBEDDINGS_PER_FEATURE),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )

    # kjt_collector = KJTAllToAll(DISTMGR.get_group())

    ref_dataloader = DataLoader(
        TorchRecCriteoLoader(
            *stage_files,    # pyre-ignore[6]
            batch_size=BATCH_SIZE,
            rank=rank,
            world_size=world_size,
            shuffle_batches=False,
            hashes=NUM_EMBEDDINGS_PER_FEATURE),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    ref_dataiter = iter(ref_dataloader)

    dense_buffer, sparse_buffer, label_buffer = [], [], []
    for idx, batch in enumerate(dataloader):
        dense, sparse, labels = batch.dense_features, batch.sparse_features, batch.labels
        dense_buffer.append(dense)
        sparse_buffer.append(sparse)
        label_buffer.append(labels)

        if (idx + 1) % world_size == 0:
            all_dense = torch.cat(dense_buffer, dim=0)
            all_labels = torch.cat(label_buffer)

            # cat sparse features
            keys, stride = sparse_buffer[0].keys(), sparse_buffer[0].stride()
            all_length_list = [kjt.lengths() for kjt in sparse_buffer]
            intermediate_all_length_list = [_length.view(-1, stride) for _length in all_length_list]
            all_length_per_key_list = [
                torch.sum(_length, dim=1).cpu().tolist() for _length in intermediate_all_length_list
            ]

            all_value_list = [kjt.values() for kjt in sparse_buffer]

            all_value_list = [
                torch.split(_values, _length_per_key)
                for _values, _length_per_key in zip(all_value_list, all_length_per_key_list)
            ]
            all_values = torch.cat([torch.cat(values_per_key) for values_per_key in zip(*all_value_list)])
            all_lengths = torch.cat(intermediate_all_length_list, dim=1).view(-1)
            all_sparse = KeyedJaggedTensor.from_lengths_sync(
                keys=keys,
                values=all_values,
                lengths=all_lengths,
                stride=batch_size,
            )

            ref_batch = next(ref_dataiter)
            ref_dense, ref_sparse, ref_labels = ref_batch.dense_features, ref_batch.sparse_features, ref_batch.labels

            assert torch.allclose(all_dense, ref_dense)
            assert torch.allclose(ref_labels, all_labels)
            all_sparse_values = (all_sparse.values().view(len(all_sparse.keys()), -1) - offsets).view(-1)
            assert torch.allclose(all_sparse_values, ref_sparse.values().long())

            dense_buffer.clear()
            sparse_buffer.clear()
            label_buffer.clear()

        if idx == world_size * 3 - 1:
            break


def run_dataloader(rank, world_size, port, stage):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, backend='nccl', host='localhost')

    dataset_dir = DATASET_DIR
    files = os.listdir(dataset_dir)

    if stage == "train":
        files = list(filter(lambda s: not is_final_day(s), files))
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = rank if stage == "val" else (rank + world_size)
        world_size = world_size * 2

    stage_files = [
        sorted(map(
            lambda x: os.path.join(dataset_dir, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]

    _test_dataloader(stage_files)

    if world_size > 1:
        _test_dist_dataloader(stage_files)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('stage', ['val', 'test'])
def test_criteo_dataloader(world_size, stage):
    run_func = partial(run_dataloader, world_size=world_size, port=free_port(), stage=stage)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_criteo_dataloader(4, 'val')