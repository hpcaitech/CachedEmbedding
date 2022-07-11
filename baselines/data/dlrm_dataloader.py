#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import List, Optional, Tuple

from torch import distributed as dist
from torch.utils.data import DataLoader
from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DAYS,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset

STAGES = ["train", "val", "test"]
KAGGLE_NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,' \
                                    '5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'  # For criteo kaggle


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
    else:
        return _get_in_memory_dataloader(args, stage)


# ============== Customize for Persia ===================
import numpy as np
from persia.embedding.data import IDTypeFeatureWithSingleID, NonIDTypeFeature, Label


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

    def __next__(self) -> Tuple[NonIDTypeFeature, IDTypeFeatureWithSingleID, Label]:
        if self.batch_index == self.num_batches:
            raise StopIteration
        if self.num_generated_batches >= 0:
            batch = self._generated_batches[self.batch_index % len(self._generated_batches)]
        else:
            batch = self._generate_batch()
        self.batch_index += 1
        return batch

    def _generate_batch(self) -> Tuple[NonIDTypeFeature, IDTypeFeatureWithSingleID, Label]:

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
