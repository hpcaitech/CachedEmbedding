#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Iterator, List, Optional
import numpy as np

from torchrec.datasets.criteo import (CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES, DAYS, BinaryCriteoUtils)
from torchrec.datasets.utils import PATH_MANAGER_KEY, Batch
from torchrec.datasets.random import RandomRecDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from iopath.common.file_io import PathManager, PathManagerFactory
from pyre_extensions import none_throws
import torch
from torch.utils.data import DataLoader, IterableDataset

from .. import ParallelMode, DISTMGR

STAGES = ["train", "val", "test"]

KAGGLE_NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,' \
                                           '27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'
KAGGLE_TOTAL_TRAINING_SAMPLES = 39291954    # 0-6 days for criteo kaggle, 45840617 samples in total


class InMemoryBinaryCriteoIterDataPipe(IterableDataset):
    """
    Datapipe designed to operate over binary (npy) versions of Criteo datasets. Loads
    the entire dataset into memory to prevent disk speed from affecting throughout. Each
    rank reads only the data for the portion of the dataset it is responsible for.

    The torchrec/datasets/scripts/npy_preproc_criteo.py script can be used to convert
    the Criteo tsv files to the npy files expected by this dataset.

    Args:
        dense_paths (List[str]): List of path strings to dense npy files.
        sparse_paths (List[str]): List of path strings to sparse npy files.
        labels_paths (List[str]): List of path strings to labels npy files.
        batch_size (int): batch size.
        rank (int): rank.
        world_size (int): world size.
        shuffle_batches (bool): Whether to shuffle batches
        hashes (Optional[int]): List of max categorical feature value for each feature.
            Length of this list should be CAT_FEATURE_COUNT.
        path_manager_key (str): Path manager key used to load from different
            filesystems.

    Example::

        template = "/home/datasets/criteo/1tb_binary/day_{}_{}.npy"
        datapipe = InMemoryBinaryCriteoIterDataPipe(
            dense_paths=[template.format(0, "dense"), template.format(1, "dense")],
            sparse_paths=[template.format(0, "sparse"), template.format(1, "sparse")],
            labels_paths=[template.format(0, "labels"), template.format(1, "labels")],
            batch_size=1024,
            rank=torch.distributed.get_rank(),
            world_size=torch.distributed.get_world_size(),
        )
        batch = next(iter(datapipe))
    """

    def __init__(
        self,
        dense_paths: List[str],
        sparse_paths: List[str],
        labels_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle_batches: bool = False,
        mmap_mode: bool = False,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.mmap_mode = mmap_mode
        self.hashes = np.array(hashes).reshape((1, CAT_FEATURE_COUNT)) if hashes is not None else None
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)
        # customization
        self.sparse_offsets = np.array([0, *np.cumsum(hashes)[:-1]]).reshape(1, -1) if hashes is not None else None

        self._load_data_for_rank()
        self.num_rows_per_file: List[int] = [a.shape[0] for a in self.dense_arrs]
        self.num_batches: int = sum(self.num_rows_per_file) // batch_size

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = CAT_FEATURE_COUNT * batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.lengths: torch.Tensor = torch.ones((self._num_ids_in_batch,), dtype=torch.int32)
        self.offsets: torch.Tensor = torch.arange(0, self._num_ids_in_batch + 1, dtype=torch.int32)
        self.stride = batch_size
        self.length_per_key: List[int] = CAT_FEATURE_COUNT * [batch_size]
        self.offset_per_key: List[int] = [batch_size * i for i in range(CAT_FEATURE_COUNT + 1)]
        self.index_per_key: Dict[str, int] = {key: i for (i, key) in enumerate(self.keys)}

    def _load_data_for_rank(self) -> None:
        file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(
            lengths=[
                BinaryCriteoUtils.get_shape_from_npy(path, path_manager_key=self.path_manager_key)[0]
                for path in self.dense_paths
            ],
            rank=self.rank,
            world_size=self.world_size,
        )

        self.dense_arrs, self.sparse_arrs, self.labels_arrs = [], [], []
        for arrs, paths in zip(
            [self.dense_arrs, self.sparse_arrs, self.labels_arrs],
            [self.dense_paths, self.sparse_paths, self.labels_paths],
        ):
            for idx, (range_left, range_right) in file_idx_to_row_range.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(
                        paths[idx],
                        range_left,
                        range_right - range_left + 1,
                        path_manager_key=self.path_manager_key,
                        mmap_mode=self.mmap_mode,
                    ))

        # When mmap_mode is enabled, the hash is applied in def __iter__, which is
        # where samples are batched during training.
        # Otherwise, the ML dataset is preloaded, and the hash is applied here in
        # the preload stage, as shown:
        if not self.mmap_mode and self.hashes is not None:
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= self.hashes
                sparse_arr += self.sparse_offsets

    def _np_arrays_to_batch(self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray) -> Batch:
        if self.shuffle_batches:
            # Shuffle all 3 in unison
            shuffler = np.random.permutation(len(dense))
            dense = dense[shuffler]
            sparse = sparse[shuffler]
            labels = labels[shuffler]

        return Batch(
            dense_features=torch.from_numpy(dense),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
        # transpose + reshape(-1) incurs an additional copy.
                values=torch.from_numpy(sparse.transpose(1, 0).reshape(-1)),
                lengths=self.lengths,
                offsets=self.offsets,
                stride=self.stride,
                length_per_key=self.length_per_key,
                offset_per_key=self.offset_per_key,
                index_per_key=self.index_per_key,
            ),
            labels=torch.from_numpy(labels.reshape(-1)),
        )

    def __iter__(self) -> Iterator[Batch]:
        # Invariant: buffer never contains more than batch_size rows.
        buffer: Optional[List[np.ndarray]] = None

        def append_to_buffer(dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray) -> None:
            nonlocal buffer
            if buffer is None:
                buffer = [dense, sparse, labels]
            else:
                for idx, arr in enumerate([dense, sparse, labels]):
                    buffer[idx] = np.concatenate((buffer[idx], arr))

        # Maintain a buffer that can contain up to batch_size rows. Fill buffer as
        # much as possible on each iteration. Only return a new batch when batch_size
        # rows are filled.
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if buffer is None else none_throws(buffer)[0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._np_arrays_to_batch(*none_throws(buffer))
                batch_idx += 1
                buffer = None
            else:
                rows_to_get = min(
                    self.batch_size - buffer_row_count,
                    self.num_rows_per_file[file_idx] - row_idx,
                )
                slice_ = slice(row_idx, row_idx + rows_to_get)

                dense_inputs = self.dense_arrs[file_idx][slice_, :]
                sparse_inputs = self.sparse_arrs[file_idx][slice_, :]
                target_labels = self.labels_arrs[file_idx][slice_, :]

                if self.mmap_mode and self.hashes is not None:
                    sparse_inputs %= self.hashes
                    sparse_inputs += self.sparse_offsets

                append_to_buffer(
                    dense_inputs,
                    sparse_inputs,
                    target_labels,
                )
                row_idx += rows_to_get

                if row_idx >= self.num_rows_per_file[file_idx]:
                    file_idx += 1
                    row_idx = 0

    def __len__(self) -> int:
        return self.num_batches


def get_dataloader(args, stage, parallel_mode: ParallelMode = ParallelMode.DEFAULT):
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (args.backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory

    if args.in_memory_binary_criteo_path is None:
        return DataLoader(
            RandomRecDataset(
                keys=DEFAULT_CAT_NAMES,
                batch_size=args.batch_size,
                hash_size=args.num_embeddings,
                hash_sizes=args.num_embeddings_per_feature if hasattr(args, "num_embeddings_per_feature") else None,
                manual_seed=args.seed if hasattr(args, "seed") else None,
                ids_per_feature=1,
                num_dense=len(DEFAULT_INT_NAMES),
                num_batches=getattr(args, f"limit_{stage}_batches")),
            batch_size=None,
            batch_sampler=None,
            pin_memory=args.pin_memory,
            num_workers=0,
        )

    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{(7 if args.kaggle else DAYS) - 1}" in s

    rank = DISTMGR.get_rank(parallel_mode)
    world_size = DISTMGR.get_world_size(parallel_mode)
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
            hashes=args.num_embeddings_per_feature),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader
