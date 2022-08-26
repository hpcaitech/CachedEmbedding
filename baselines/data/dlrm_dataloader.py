#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List, Optional, Tuple, Dict
import glob
import numpy as np

import torch
from torch import distributed as dist
from torch.utils.data import DataLoader, IterableDataset
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DEFAULT_LABEL_NAME,
    DAYS,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import Batch
from petastorm import make_batch_reader
from pyarrow.parquet import ParquetDataset
from .avazu import AvazuIterDataPipe

STAGES = ["train", "val", "test"]
KAGGLE_NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,' \
                                    '5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'  # For criteo kaggle
KAGGLE_TOTAL_TRAINING_SAMPLES = 39_291_954    # 0-6 days for criteo kaggle, 45,840,617 samples in total
TERABYTE_NUM_EMBEDDINGS_PER_FEATURE = "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10," \
                                      "2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"


def _get_random_dataloader(args: argparse.Namespace,) -> DataLoader:
    return DataLoader(
        RandomRecDataset(
            keys=DEFAULT_CAT_NAMES,
            batch_size=args.batch_size,
            hash_size=args.num_embeddings,
            hash_sizes=args.num_embeddings_per_feature if hasattr(args, "num_embeddings_per_feature") else None,
            manual_seed=args.seed if hasattr(args, "seed") else None,
            ids_per_feature=1,
            num_dense=len(DEFAULT_INT_NAMES),
        ),
        batch_size=None,
        batch_sampler=None,
        pin_memory=args.pin_memory,
        num_workers=0,
    )


def _get_in_memory_dataloader(
    args: argparse.Namespace,
    stage: str,
) -> DataLoader:
    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{(7 if args.kaggle else DAYS) - 1}" in s

    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = (dist.get_rank() if stage == "val" else dist.get_rank() + dist.get_world_size())
        world_size = dist.get_world_size() * 2

    stage_files: List[List[str]] = [
        sorted(map(
            lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            *stage_files,    # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=args.shuffle_batches,
            hashes=args.num_embeddings_per_feature if args.num_embeddings is None else
            ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def get_avazu_data_loader(args, stage):
    files = os.listdir(args.in_memory_binary_criteo_path)

    if stage == "train":
        files = list(filter(lambda s: "train" in s, files))
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(lambda s: "train" not in s, files))
        rank = (dist.get_rank() if stage == "val" else dist.get_rank() + dist.get_world_size())
        world_size = dist.get_world_size() * 2

    stage_files: List[List[str]] = [
        sorted(map(
            lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "label"]
    ]

    dataloader = DataLoader(
        AvazuIterDataPipe(
            *stage_files,    # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=args.shuffle_batches,
            hashes=args.num_embeddings_per_feature if args.num_embeddings is None else
            ([args.num_embeddings] * CAT_FEATURE_COUNT),
        ),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


class PetastormDataReader(IterableDataset):
    """
    This is a compromise solution for the criteo terabyte dataset,
    please see the solution 3 in: https://github.com/uber/petastorm/issues/508

    Basically, the dataloader in each rank extracts random samples from the whole dataset in the training stage
    in which the batches in each rank are not guaranteed to be unique.
    In the validation stage, all the samples are evaluated in each rank,
    so that each rank contains the correct result
    """

    def __init__(self,
                 paths,
                 batch_size,
                 rank=None,
                 world_size=None,
                 shuffle_batches=False,
                 hashes=None,
                 seed=1024,
                 drop_last=True):
        self.dataset = ParquetDataset(paths, use_legacy_dataset=False)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.hashes = np.array(hashes).reshape((1, CAT_FEATURE_COUNT)) if hashes is not None else None

        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones((self._num_ids_in_batch,), dtype=torch.int32)
        self.offsets: torch.Tensor = torch.arange(0, self._num_ids_in_batch + 1, dtype=torch.int32)
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]
        self.index_per_key: Dict[str, int] = {key: i for (i, key) in enumerate(self.keys)}
        self.seed = seed

        self.drop_last = drop_last
        if drop_last:
            self.num_batches = sum([fragment.metadata.num_rows for fragment in self.dataset.fragments
                                   ]) // self.batch_size
        else:
            self.num_batches = (sum([fragment.metadata.num_rows
                                     for fragment in self.dataset.fragments]) + self.batch_size - 1) // self.batch_size
        if self.world_size is not None:
            self.num_batches = self.num_batches // world_size

    def __iter__(self):
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(_dense: np.ndarray, _sparse: np.ndarray, _labels: np.ndarray) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [_dense, _sparse, _labels]
            else:
                buffer[0] = np.concatenate([buffer[0], _dense], axis=0)
                buffer[1] = np.concatenate([buffer[1], _sparse], axis=1)
                buffer[2] = np.concatenate([buffer[2], _labels], axis=0)

        with make_batch_reader(
                list(map(lambda x: "file://" + x, self.dataset.files)),
                num_epochs=1,
                workers_count=1,    # for reproducibility
        ) as reader:
            # note that `batch` here is just a bunch of samples read by petastorm instead of `batch` consumed by models
            for _ in range(self.num_batches):
                batch = next(reader)
                labels = getattr(batch, DEFAULT_LABEL_NAME)
                sparse = np.concatenate([getattr(batch, col_name).reshape(1, -1) for col_name in DEFAULT_CAT_NAMES],
                                        axis=0)
                dense = np.concatenate([getattr(batch, col_name).reshape(-1, 1) for col_name in DEFAULT_INT_NAMES],
                                       axis=1)
                start_idx = 0
                while start_idx < dense.shape[0]:
                    buffer_size = 0 if buffer is None else buffer[0].shape[0]
                    if buffer_size == self.batch_size:
                        yield self._batch_ndarray(*buffer)
                        buffer = None
                    else:
                        rows_to_get = min(self.batch_size - buffer_size, dense.shape[0] - start_idx)
                        label_chunk = labels[start_idx:start_idx + rows_to_get]
                        sparse_chunk = sparse[:, start_idx:start_idx + rows_to_get]
                        dense_chunk = dense[start_idx:start_idx + rows_to_get, :]
                        append_to_buffer(dense_chunk, sparse_chunk, label_chunk)
                        start_idx += rows_to_get
        if buffer is not None and not self.drop_last:
            yield self._batch_ndarray(*buffer)

    def _batch_ndarray(self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray):
        if self.shuffle_batches:
            # Shuffle all 3 in unison
            shuffler = np.random.permutation(len(dense))
            dense = dense[shuffler]
            sparse = sparse[:, shuffler]
            labels = labels[shuffler]

        return Batch(
            dense_features=torch.from_numpy(dense),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                values=torch.from_numpy(sparse.reshape(-1)),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1)),
        )

    def __len__(self):
        return self.num_batches


def _get_petastorm_dataloader(args, stage):
    if stage == "train":
        data_split = "train"
    elif stage == "val":
        data_split = "validation"
    else:
        data_split = "test"

    file_num = len(glob.glob(os.path.join(args.in_memory_binary_criteo_path, data_split, "*.parquet")))
    files = [os.path.join(args.in_memory_binary_criteo_path, data_split, f"part_{i}.parquet") for i in range(file_num)]

    dataloader = DataLoader(PetastormDataReader(files,
                                                args.batch_size,
                                                rank=dist.get_rank() if stage == "train" else None,
                                                world_size=dist.get_world_size() if stage == "train" else None,
                                                hashes=args.num_embeddings_per_feature),
                            batch_size=None,
                            pin_memory=False,
                            collate_fn=lambda x: x,
                            num_workers=0)

    return dataloader


def get_dataloader(args: argparse.Namespace, backend: str, stage: str) -> DataLoader:
    """
    Gets desired dataloader from dlrm_main command line options. Currently, this
    function is able to return either a DataLoader wrapped around a RandomRecDataset or
    a Dataloader wrapped around an InMemoryBinaryCriteoIterDataPipe.

    Args:
        args (argparse.Namespace): Command line options supplied to dlrm_main.py's main
            function.
        backend (str): "nccl" or "gloo".
        stage (str): "train", "val", or "test".

    Returns:
        dataloader (DataLoader): PyTorch dataloader for the specified options.

    """
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = ((backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory)

    if (not hasattr(args, "in_memory_binary_criteo_path") or args.in_memory_binary_criteo_path is None):
        return _get_random_dataloader(args)
    elif "criteo" in args.in_memory_binary_criteo_path:
        if args.kaggle:
            return _get_in_memory_dataloader(args, stage)
        else:
            return _get_petastorm_dataloader(args, stage)
    elif "avazu" in args.in_memory_binary_criteo_path:
        return get_avazu_data_loader(args, stage)


# ============== Customize for Persia ===================

try:
    from persia.embedding.data import IDTypeFeatureWithSingleID, NonIDTypeFeature, Label
except ImportError:
    pass


class PersiaDataLoader:

    def __init__(self, dense_features, sparse_features, labels, batch_size, skip_last_batch=True):
        self.dense_features = dense_features
        self.sparse_features = sparse_features
        self.labels = labels
        self.batch_size = batch_size
        self.skip_last_batch = skip_last_batch

        dataset_size = labels.shape[0]
        loader_size = (dataset_size - 1) // batch_size + 1
        if skip_last_batch:
            loader_size = loader_size - 1
        self.loader_size = loader_size

    def __iter__(self):
        dataset_size = self.labels.shape[0]
        for start in range(0, dataset_size, self.batch_size):
            end = min(start + self.batch_size, dataset_size)
            if end == dataset_size and self.skip_last_batch:
                continue

            dense_features = NonIDTypeFeature(self.dense_features[start:end])
            labels = Label(self.labels[start:end])
            sparse_features = []
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES):
                sparse_features.append(
                    IDTypeFeatureWithSingleID(
                        feature_name, np.ascontiguousarray(self.sparse_features[start:end, feature_idx],
                                                           dtype=np.uint64)))

            yield dense_features, sparse_features, labels

    def __len__(self):
        return self.loader_size


class RandomDataLoader:
    """
    Adapted from torchrec.datasets.random.RandomRecDataset
    """
    generator: Optional[np.random.Generator]

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_sizes: List[int],
        ids_per_features: List[int],
        num_dense: int,
        manual_seed: Optional[int] = None,
        num_generated_batches: int = 10,
        num_batches: Optional[int] = None,
    ) -> None:

        self.keys = keys
        self.keys_length: int = len(keys)
        self.batch_size = batch_size
        self.hash_sizes = hash_sizes
        self.ids_per_features = ids_per_features
        self.num_dense = num_dense
        self.num_batches = num_batches
        self.num_generated_batches = num_generated_batches

        assert len(keys) == len(hash_sizes), f"num keys: {len(keys)}, num features: {len(hash_sizes)}"
        if manual_seed is not None:
            self.generator = np.random.default_rng(seed=manual_seed)
        else:
            self.generator = np.random.default_rng()

        self._generated_batches: List[Tuple[NonIDTypeFeature, IDTypeFeatureWithSingleID,
                                            Label]] = [self._generate_batch()] * num_generated_batches
        self.batch_index = 0

    def __iter__(self) -> "RandomDataLoader":
        self.batch_index = 0
        return self

    def __next__(self):
        if self.batch_index == self.num_batches:
            raise StopIteration
        if self.num_generated_batches >= 0:
            batch = self._generated_batches[self.batch_index % len(self._generated_batches)]
        else:
            batch = self._generate_batch()
        self.batch_index += 1
        return batch

    def _generate_batch(self):

        sparse_features = []
        for key_idx, key_name in enumerate(self.keys):
            hash_size = self.hash_sizes[key_idx]
            num_ids_in_batch = self.ids_per_features[key_idx]
            sparse_features.append(
                IDTypeFeatureWithSingleID(
                    key_name, self.generator.integers(low=0, high=hash_size, size=(self.batch_size,),
                                                      dtype=np.uint64)),)

        dense_features = NonIDTypeFeature(
            self.generator.normal(size=(self.batch_size, self.num_dense)).astype(np.float32))
        labels = Label(self.generator.integers(low=0, high=2, size=(self.batch_size, 1)))
        return dense_features, sparse_features, labels


def get_persia_dataloader(stage,
                          dataset_dir,
                          batch_size,
                          is_kaggle=True,
                          num_embeddings_per_feature=None,
                          seed=None,
                          num_batches=None):
    if dataset_dir is None:
        num_embeddings_per_feature = list(map(int, num_embeddings_per_feature.split(',')))
        if len(num_embeddings_per_feature) == 1:
            num_embeddings_per_feature = [num_embeddings_per_feature[0]] * len(DEFAULT_CAT_NAMES)
        return RandomDataLoader(
            keys=DEFAULT_CAT_NAMES,
            batch_size=batch_size,
            hash_sizes=num_embeddings_per_feature,
            ids_per_features=[1] * len(DEFAULT_CAT_NAMES),
            num_dense=len(DEFAULT_INT_NAMES),
            manual_seed=seed,
            num_batches=num_batches,
        )

    files = os.listdir(dataset_dir)

    def is_final_day(s):
        return f"day_{(7 if is_kaggle else DAYS) - 1}" in s

    if stage == "train":
        files = list(filter(lambda s: not is_final_day(s), files))
    else:
        files = list(filter(is_final_day, files))

    stage_files: List[List[str]] = [
        sorted(map(
            lambda x: os.path.join(dataset_dir, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]

    dense_features, sparse_features, labels = (
        np.vstack([np.load(path) for path in type_files]) for type_files in stage_files)

    num_samples = dense_features.shape[0]
    if stage == "val":
        dense_features = dense_features[:num_samples // 2]
        sparse_features = sparse_features[:num_samples // 2]
        labels = labels[:num_samples // 2]
    elif stage == "test":
        dense_features = dense_features[num_samples // 2:]
        sparse_features = sparse_features[num_samples // 2:]
        labels = labels[num_samples // 2:]

    hashes = np.array(list(map(int, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(',')))).reshape(1, CAT_FEATURE_COUNT)
    sparse_features %= hashes
    sparse_features = sparse_features.astype(np.uint64)
    return PersiaDataLoader(dense_features, sparse_features, labels, batch_size)


if __name__ == "__main__":
    loader = get_persia_dataloader('test', '../../criteo_kaggle', 16384)
    print(len(loader))
