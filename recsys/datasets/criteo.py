#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import Dict, Iterator, List, Optional
import numpy as np
import glob

from torchrec.datasets.criteo import (CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES, DEFAULT_LABEL_NAME, DAYS,
                                      BinaryCriteoUtils)
from torchrec.datasets.utils import PATH_MANAGER_KEY, Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from iopath.common.file_io import PathManager, PathManagerFactory
from pyre_extensions import none_throws
import torch
from torch.utils.data import DataLoader, IterableDataset
from petastorm import make_batch_reader
from pyarrow.parquet import ParquetDataset

from .feature_counter import GlobalFeatureCounter, PetastormCounter

STAGES = ["train", "val", "test"]

# 177,944,275 in total
NUM_EMBEDDINGS_PER_FEATURE = "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267," \
                             "128,4,974,14,48937457,11316796,40094537,452104,12606,104,35"
# 33,762,577 in total
KAGGLE_NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,' \
                                           '27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'
KAGGLE_TOTAL_TRAINING_SAMPLES = 39_291_954    # 0-6 days for criteo kaggle, 45,840,617 samples in total


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
        assigned_tables (Optional[List[int]]): For tablewise mode. sparse features(tables) are numbered as 0, 1, 2, ..., n. 
                        appending a number to assigned_tables means the coresponding table is assinged to this loader.
                        iff a table is assigned to this loader should it be loaded into the batch.

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
        assigned_tables: Optional[List[int]] = None
    ) -> None:
        if assigned_tables is not None:
            # tablewise mode
            self.assigned_tables = np.array(assigned_tables)
        else:
            # full table mode
            self.assigned_tables = np.arange(CAT_FEATURE_COUNT)
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.labels_paths = labels_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.mmap_mode = mmap_mode
        # self.hashes = np.array(hashes).reshape((1, CAT_FEATURE_COUNT)) if hashes is not None else None
        if hashes is not None:
            self.hashes = []
            for i, length in enumerate(hashes):
                if i in self.assigned_tables:
                    self.hashes.append(length)
            self.hashes = np.array(self.hashes).reshape(1,-1)
        else:
            self.hashes = None
        self.path_manager_key = path_manager_key
        self.path_manager: PathManager = PathManagerFactory().get(path_manager_key)
        
        # customization
        self.sparse_offsets = np.array([0, *np.cumsum(self.hashes)[:-1]], dtype=np.int64).reshape(
                    1, -1) if self.hashes is not None else None
        self._load_data_for_rank()
        self.num_rows_per_file: List[int] = [a.shape[0] for a in self.dense_arrs]
        self.num_batches: int = sum(self.num_rows_per_file) // batch_size

        # These values are the same for the KeyedJaggedTensors in all batches, so they
        # are computed once here. This avoids extra work from the KeyedJaggedTensor sync
        # functions.
        self._num_ids_in_batch: int = len(self.assigned_tables) * batch_size
        self.keys: List[str] = [DEFAULT_CAT_NAMES[i] for i in self.assigned_tables]
        self.lengths: torch.Tensor = torch.ones((self._num_ids_in_batch,), dtype=torch.int32)
        self.offsets: torch.Tensor = torch.arange(0, self._num_ids_in_batch + 1, dtype=torch.int32)
        self.stride = batch_size
        self.length_per_key: List[int] = len(self.assigned_tables) * [batch_size]
        self.offset_per_key: List[int] = [batch_size * i for i in range(len(self.assigned_tables) + 1)]
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
        for _dtype, arrs, paths in zip(
            [np.float32, np.int64, np.int32],    # TODO: data type interface
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
                    ).astype(_dtype))

        # When mmap_mode is enabled, the hash is applied in def __iter__, which is
        # where samples are batched during training.
        # Otherwise, the ML dataset is preloaded, and the hash is applied here in
        # the preload stage, as shown:
        expand_hashes = np.ones(CAT_FEATURE_COUNT, dtype=np.int64).reshape(1,-1)
        expand_sparse_offsets = np.ones(CAT_FEATURE_COUNT, dtype=np.int64).reshape(1,-1)
        for i, table in enumerate(self.assigned_tables):
            expand_hashes[0,table] = self.hashes[0,i]
            expand_sparse_offsets[0,table] = self.sparse_offsets[0,i]
        if not self.mmap_mode and self.hashes is not None:
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= expand_hashes
                sparse_arr += expand_sparse_offsets

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
                sparse_inputs = self.sparse_arrs[file_idx][slice_, :].take(self.assigned_tables, -1)
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


class PetastormDataReader(IterableDataset):

    def __init__(self,
                 paths,
                 batch_size,
                 rank=None,
                 world_size=None,
                 shuffle_batches=False,
                 hashes=None,
                 seed=1024,
                 drop_last=True,
                 assigned_tables=None):
        
        if assigned_tables is not None:
            # tablewise mode
            self.assigned_tables = np.array(assigned_tables)
        else:
            # full table mode
            self.assigned_tables = np.arange(CAT_FEATURE_COUNT)
        self.dataset = ParquetDataset(paths, use_legacy_dataset=False)
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        if hashes is not None:
            self.hashes = []
            for i, length in enumerate(hashes):
                if i in self.assigned_tables:
                    self.hashes.append(length)
            self.hashes = np.array(self.hashes).reshape(1, -1)
        else:
            self.hashes = None
        self.sparse_offsets = np.array([0, *np.cumsum(self.hashes)[:-1]], dtype=np.int64).reshape(
                    -1, 1) if self.hashes is not None else None

        self._num_ids_in_batch: int = len(self.assigned_tables) * batch_size
        self.keys: List[str] = [DEFAULT_CAT_NAMES[i] for i in self.assigned_tables]
        self.lengths: torch.Tensor = torch.ones((self._num_ids_in_batch,), dtype=torch.int32)
        self.offsets: torch.Tensor = torch.arange(0, self._num_ids_in_batch + 1, dtype=torch.int32)
        self.stride = batch_size
        self.length_per_key: List[int] = len(self.assigned_tables) * [batch_size]
        self.offset_per_key: List[int] = [batch_size * i for i in range(len(self.assigned_tables) + 1)]
        self.index_per_key: Dict[str, int] = {key: i for (i, key) in enumerate(self.keys)}
        self.seed = seed
        self.epoch = 0

        self.drop_last = drop_last
        if drop_last:
            self.num_batches = sum([fragment.metadata.num_rows for fragment in self.dataset.fragments
                                   ]) // self.batch_size
        else:
            self.num_batches = (sum([fragment.metadata.num_rows
                                     for fragment in self.dataset.fragments]) + self.batch_size - 1) // self.batch_size

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

        random.seed(self.seed + self.epoch)    # for sync RNG inside the petastorm reader
        self.epoch += 1
        with make_batch_reader(
                list(map(lambda x: "file://" + x, self.dataset.files)),
                num_epochs=1,
                workers_count=1,    # for reproducibility
        ) as reader:
            # note that `batch` here is just a bunch of samples read by petastorm instead of `batch` consumed by models
            for batch in reader:
                labels = getattr(batch, DEFAULT_LABEL_NAME)
                sparse = np.concatenate([getattr(batch, col_name).reshape(1, -1) for col_name in self.keys],
                                        axis=0)
                if self.sparse_offsets is not None:
                    sparse = sparse + self.sparse_offsets
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


def _get_kaggle_dataloader(args, stage, rank, world_size, assigned_tables=None):
    files = os.listdir(args.dataset_dir)

    def is_final_day(s: str) -> bool:
        return f"day_{(7 if 'kaggle' in args.dataset_dir else DAYS) - 1}" in s

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
            lambda x: os.path.join(args.dataset_dir, x),
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
            hashes=args.num_embeddings_per_feature,
            assigned_tables=assigned_tables),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader


def _get_terabyte_dataloader(args, stage, rank, world_size, assigned_tables=None):
    # TODO: replace the data_split with stage
    if stage == "train":
        data_split = "train"
    elif stage == "val":
        data_split = "validation"
    else:
        data_split = "test"

    if world_size > 1 or rank != 0:
        raise RuntimeError("We do not support distributed dataloader currently.")

    file_num = len(glob.glob(os.path.join(args.dataset_dir, data_split, "*.parquet")))
    files = [os.path.join(args.dataset_dir, data_split, f"part_{i}.parquet") for i in range(file_num)]

    dataloader = DataLoader(PetastormDataReader(files,
                                                args.batch_size,
                                                rank=None,
                                                world_size=None,
                                                shuffle_batches=stage == "train",
                                                hashes=args.num_embeddings_per_feature,
                                                seed=args.seed,
                                                assigned_tables=assigned_tables),
                            batch_size=None,
                            pin_memory=False,
                            collate_fn=lambda x: x,
                            num_workers=0)

    return dataloader


def get_dataloader(args, stage, rank, world_size, assigned_tables = None):
    '''
    if assigned_tables is not None, the dataloader is in tablewise mode.
    '''
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    if "kaggle" in args.dataset_dir:
        return _get_kaggle_dataloader(args, stage, rank, world_size, assigned_tables)
    else:
        return _get_terabyte_dataloader(args, stage, rank, world_size, assigned_tables)


def get_id_freq_map(path):
    checkpoint_path = os.path.join(path, "id_freq_map.pt")
    if os.path.exists(checkpoint_path):
        id_freq_map = torch.load(checkpoint_path)
        return id_freq_map

    if 'kaggle' not in path:
        file_num = len(glob.glob(os.path.join(path, "train", "*.parquet")))
        files = [os.path.join(path, "train", f"part_{i}.parquet") for i in range(file_num)]
        feature_count = PetastormCounter(files,
                                         list(map(int, NUM_EMBEDDINGS_PER_FEATURE.split(','))),
                                         subsample_fraction=0.1)
        id_freq_map = feature_count.compute()
    else:
        files = os.listdir(path)
        sparse_files = list(filter(lambda s: 'sparse' in s, files))
        sparse_files = [os.path.join(path, _f) for _f in sparse_files]

        feature_count = GlobalFeatureCounter(sparse_files, list(map(int, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(','))))
        id_freq_map = feature_count.compute()

    id_freq_map = torch.from_numpy(id_freq_map)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(id_freq_map, checkpoint_path)

    return id_freq_map
