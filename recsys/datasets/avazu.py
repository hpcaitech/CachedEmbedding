import os

import numpy as np

import torch
import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe, IterableDataset, DataLoader
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.utils import LoadFiles, ReadLinesFromCSV, PATH_MANAGER_KEY, Batch
from torchrec.datasets.criteo import BinaryCriteoUtils

from .feature_counter import GlobalFeatureCounter

CAT_FEATURE_COUNT = 13
INT_FEATURE_COUNT = 8
DAYS = 10
DEFAULT_LABEL_NAME = "click"
DEFAULT_CAT_NAMES = [
    'C1',
    'banner_pos',
    'site_id',
    'site_domain',
    'site_category',
    'app_id',
    'app_domain',
    'app_category',
    'device_id',
    'device_ip',
    'device_model',
    'device_type',
    'device_conn_type',
]
DEFAULT_INT_NAMES = ['C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21']
NUM_EMBEDDINGS_PER_FEATURE = '7,7,4737,7745,26,8552,559,36,2686408,6729486,8251,5,4'    # 9445823 in total
TOTAL_TRAINING_SAMPLES = 36_386_071    # 90% sample in train, 40428967 in total


def _default_row_mapper(row):
    _label = row[1]
    _sparse = row[3:5]
    for i in range(5, 14):    # 9
        try:
            _c = int(row[i], 16)
        except ValueError:
            _c = 0
        _sparse.append(_c)
    _sparse += row[14:24]

    return _sparse, _label


class AvazuIterDataPipe(IterDataPipe):

    def __init__(self, path, row_mapper=_default_row_mapper):
        self.path = path
        self.row_mapper = row_mapper

    def __iter__(self):
        """
        iterate over the data file, and apply the transform row_mapper to each row
        """
        datapipe = LoadFiles([self.path], mode='r', path_manager_key='avazu')
        datapipe = ReadLinesFromCSV(datapipe, delimiter=',', skip_first_line=True)
        if self.row_mapper is not None:
            datapipe = dp.iter.Mapper(datapipe, self.row_mapper)
        yield from datapipe


class InMemoryAvazuIterDataPipe(IterableDataset):

    def __init__(self,
                 dense_paths,
                 sparse_paths,
                 label_paths,
                 batch_size,
                 rank,
                 world_size,
                 shuffle_batches=False,
                 mmap_mode=False,
                 hashes=None,
                 path_manager_key=PATH_MANAGER_KEY,
                 assigned_tables = None):
        if assigned_tables is not None:
            # tablewise mode
            self.assigned_tables = np.array(assigned_tables)
        else:
            # full table mode
            self.assigned_tables = np.arange(CAT_FEATURE_COUNT)
            
        self.dense_paths = dense_paths
        self.sparse_paths = sparse_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.mmap_mode = mmap_mode
        if hashes is not None:
            self.hashes = []
            for i, length in enumerate(hashes):
                if i in self.assigned_tables:
                    self.hashes.append(length)
            self.hashes = np.array(self.hashes).reshape(1, -1)
        else:
            self.hashes = None
        self.path_manager_key = path_manager_key

        self.sparse_offsets = np.array([0, *np.cumsum(self.hashes)[:-1]], dtype=np.int64).reshape(
            1, -1) if self.hashes is not None else None

        self._load_data()
        self.num_rows_per_file = [a.shape[0] for a in self.dense_arrs]
        self.num_batches: int = sum(self.num_rows_per_file) // batch_size

        self._num_ids_in_batch: int = len(self.assigned_tables) * batch_size
        self.keys = [DEFAULT_CAT_NAMES[i] for i in self.assigned_tables]
        self.lengths = torch.ones((self._num_ids_in_batch,), dtype=torch.int32)
        self.offsets = torch.arange(0, self._num_ids_in_batch + 1, dtype=torch.int32)
        self.stride = batch_size
        self.length_per_key = len(self.assigned_tables) * [batch_size]
        self.offset_per_key = [batch_size * i for i in range(len(self.assigned_tables) + 1)]
        self.index_per_key = {key: i for (i, key) in enumerate(self.keys)}

    def _load_data(self):
        file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(lengths=[
            BinaryCriteoUtils.get_shape_from_npy(path, path_manager_key=self.path_manager_key)[0]
            for path in self.sparse_paths
        ],
                                                                            rank=self.rank,
                                                                            world_size=self.world_size)
        self.dense_arrs, self.sparse_arrs, self.labels_arrs = [], [], []
        for _dtype, arrs, paths in zip([np.float32, np.int64, np.int32],
                                       [self.dense_arrs, self.sparse_arrs, self.labels_arrs],
                                       [self.dense_paths, self.sparse_paths, self.label_paths]):
            for idx, (range_left, range_right) in file_idx_to_row_range.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(paths[idx],
                                                     range_left,
                                                     range_right - range_left + 1,
                                                     path_manager_key=self.path_manager_key,
                                                     mmap_mode=self.mmap_mode).astype(_dtype))
        expand_hashes = np.ones(CAT_FEATURE_COUNT, dtype=np.int64).reshape(1, -1)
        expand_sparse_offsets = np.ones(CAT_FEATURE_COUNT, dtype=np.int64).reshape(1, -1)
        for i, table in enumerate(self.assigned_tables):
            expand_hashes[0, table] = self.hashes[0, i]
            expand_sparse_offsets[0, table] = self.sparse_offsets[0, i]
        if not self.mmap_mode and self.hashes is not None:
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= expand_hashes
                sparse_arr += expand_sparse_offsets

    def __iter__(self):
        buffer = None

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
            buffer_row_count = 0 if buffer is None else buffer[0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._np_arrays_to_batch(*buffer)
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

    def _np_arrays_to_batch(self, dense: np.ndarray, sparse: np.ndarray, labels: np.ndarray) -> Batch:
        if self.shuffle_batches:
            # Shuffle all 3 in unison
            shuffler = np.random.permutation(sparse.shape[0])
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

    def __len__(self) -> int:
        return self.num_batches


def get_dataloader(args, stage, rank, world_size,  assigned_tables = None):
    stage = stage.lower()

    files = os.listdir(args.dataset_dir)

    if stage == 'train':
        files = list(filter(lambda s: 'train' in s, files))
    else:
        files = list(filter(lambda s: 'train' not in s, files))
        rank = rank if stage == "val" else (rank + world_size)
        world_size = world_size * 2

    stage_files = [
        sorted(map(
            lambda s: os.path.join(args.dataset_dir, s),
            filter(lambda _f: kind in _f, files),
        )) for kind in ["dense", "sparse", "label"]
    ]

    dataloader = DataLoader(
        InMemoryAvazuIterDataPipe(*stage_files,
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


def get_id_freq_map(path):
    files = os.listdir(path)
    files = list(filter(lambda s: "sparse" in s, files))
    files = [os.path.join(path, _f) for _f in files]

    feature_count = GlobalFeatureCounter(files, list(map(int, NUM_EMBEDDINGS_PER_FEATURE.split(','))))
    id_freq_map = torch.from_numpy(feature_count.compute())
    return id_freq_map
