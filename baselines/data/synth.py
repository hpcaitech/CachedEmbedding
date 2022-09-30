import argparse
import torch
from torch import distributed as dist
import numpy as np
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from colossalai.nn.parallel.layers.cache_embedding import FreqAwareEmbeddingBag
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from fbgemm_gpu.split_table_batched_embeddings_ops import SplitTableBatchedEmbeddingBagsCodegen, EmbeddingLocation, ComputeDevice, CacheAlgorithm
import time
import numpy as np
import torch
from torch.utils.data import IterableDataset
from torchrec.datasets.utils import PATH_MANAGER_KEY, Batch
from torchrec.datasets.criteo import BinaryCriteoUtils
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torch.utils.data import DataLoader, IterableDataset
import os

CHOSEN_TABLES = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 18, 22, 27, 28]
NUM_EMBEDDINGS_PER_FEATURE = '8015999, 9997799, 6138289, 21886, 204008, 6148, 282795, \
    1316, 3639992, 319, 3394206, 12203324, 4091851, 11641, 4657566'
CAT_FEATURE_COUNT = len(CHOSEN_TABLES)
INT_FEATURE_COUNT = 1
DEFAULT_LABEL_NAME = "click"
DEFAULT_INT_NAMES = ['rand_dense']
BATCH_SIZE = 65536 # batch_size of one file
DEFAULT_CAT_NAMES = ["cat_{}".format(i) for i in range(len(CHOSEN_TABLES))]

class SynthIterDataPipe(IterableDataset):
    def __init__(
        self,
        sparse_paths: List[str],
        batch_size: int,
        rank: int,
        world_size: int,
        shuffle_batches: bool = False,
        mmap_mode: bool = False,
        hashes: Optional[List[int]] = None,
        path_manager_key: str = PATH_MANAGER_KEY,
    ) -> None:
        self.sparse_paths = sparse_paths,
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.mmap_mode = mmap_mode
        self.hashes = hashes
        self.path_manager_key = path_manager_key
        
        self.indices_per_table_per_file = []
        self.offsets_per_table_per_file = []
        self.lengths_per_table_per_file = []
        self.num_rows_per_file = []
        for file in self.sparse_paths[0]:
            print("load file: ", file)
            indices_per_table, offsets_per_table, lengths_per_table = self._load_single_file(file)
            self.indices_per_table_per_file.append(indices_per_table)
            self.offsets_per_table_per_file.append(offsets_per_table)
            self.lengths_per_table_per_file.append(lengths_per_table)
            self.num_rows_per_file.append(offsets_per_table[0].shape[0])
        self.num_batches = sum(self.num_rows_per_file) // self.batch_size
        self.keys: List[str] = DEFAULT_CAT_NAMES
        self.stride = batch_size
        
    def __iter__(self) -> Iterator[Batch]:
        buffer = None
        # buffer structure:
        '''
        buffer[0]: List of sparse_indices per table
        buffer[1]: List of sparse_lengths per table
        '''
        def append_to_buffer(sparse_indices: List[torch.Tensor], sparse_lengths: List[torch.Tensor]):
            nonlocal buffer
            if buffer is None:
                buffer = [sparse_indices, sparse_lengths]
            else:
                for tb_idx, (sparse_indices_table, sparse_lengths_table) in enumerate(zip(sparse_indices, sparse_lengths)):
                    buffer[0][tb_idx] = torch.cat((buffer[0][tb_idx], sparse_indices_table))
                    buffer[1][tb_idx] = torch.cat((buffer[1][tb_idx], sparse_lengths_table))
                    
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if buffer is None else buffer[1][0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._make_batch(*buffer)
                batch_idx += 1
                buffer = None
            else:
                rows_to_get = min(
                    self.batch_size - buffer_row_count,
                    BATCH_SIZE - row_idx
                )
                slice_ = slice(row_idx, row_idx + rows_to_get)
                sparse_indices, sparse_lengths = self._load_random_batch(self.indices_per_table_per_file[file_idx],
                                                                         self.offsets_per_table_per_file[file_idx],
                                                                         self.lengths_per_table_per_file[file_idx],
                                                                         slice_
                                                                         )
                append_to_buffer(sparse_indices, sparse_lengths)
                row_idx += rows_to_get
                if row_idx >= BATCH_SIZE:
                    file_idx += 1
                    row_idx = 0
                    
    def __len__(self) -> int:
        return self.num_batches
    
    def _load_single_file(self, file_path):
        # TODO: shard loading
        # file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(
        #     lengths=[BATCH_SIZE],
        #     rank=self.rank,
        #     world_size=self.world_size
        # )
        # for _,(range_left, range_right) in file_idx_to_row_range.items():
        #     rank_range_left = range_left
        #     rank_range_right = range_right
        rank_range_left = 0
        rank_range_right = 65535
        indices, offsets, lengths = torch.load(file_path)
        indices = indices.int()
        offsets = offsets.int()
        lengths = lengths.int()
        indices_per_table = []
        offsets_per_table = []
        lengths_per_table = []
        for i in CHOSEN_TABLES:
            start_pos = offsets[i * BATCH_SIZE + rank_range_left]
            end_pos = offsets[i * BATCH_SIZE + rank_range_right + 1]
            part = indices[start_pos:end_pos]
            indices_per_table.append(part)
            lengths_per_table.append(lengths[i][rank_range_left:rank_range_right + 1])
            offsets_per_table.append(torch.cumsum(
                torch.cat((torch.tensor([0]), lengths[i][rank_range_left:rank_range_right + 1])), 0
            ))
        return indices_per_table, offsets_per_table, lengths_per_table

    def _load_random_batch(self, indices_per_table, offsets_per_table, lengths_per_table, choose):
        chosen_indices_list = []
        chosen_lengths_list = []
        # choose = torch.randint(0, offsets_per_table[0].shape[0] - 1, (self.batch_size,))
        for indices, offsets, lengths in zip(indices_per_table, offsets_per_table, lengths_per_table):
            chosen_lengths_list.append(lengths[choose])
            start_list = offsets[choose]
            end_list = offsets[1:][choose]
            chosen_indices_atoms = []
            for start, end in zip(start_list, end_list):
                chosen_indices_atoms.append(indices[start: end])
            chosen_indices_list.append(torch.cat(chosen_indices_atoms, 0))
        return chosen_indices_list, chosen_lengths_list
    
    def _make_batch(self, chosen_indices_list, chosen_lengths_list):
        batch_size = chosen_lengths_list[0].shape[0]
        ret =  Batch(
            dense_features=torch.rand(batch_size,1),
            sparse_features=KeyedJaggedTensor(
                keys=self.keys,
                values=torch.cat(chosen_indices_list),
                lengths=torch.cat(chosen_lengths_list),
                stride=batch_size,
            ),
            labels=torch.randint(2, (batch_size,))
        )
        return ret
def get_synth_data_loader(
        args: argparse.Namespace,
        stage: str) -> DataLoader:
    files = os.listdir(args.in_memory_binary_criteo_path)
    files = filter(lambda s: "fbgemm_t856_bs65536" in s, files)
    files = [os.path.join(args.in_memory_binary_criteo_path, x) for x in files]
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if stage == "train":
        dataloader = DataLoader(
            SynthIterDataPipe(
                files,
                batch_size=args.batch_size,
                rank=rank,
                world_size=world_size,
                shuffle_batches=args.shuffle_batches,
                hashes = None
            ),
            batch_size=None,
            pin_memory=args.pin_memory,
            collate_fn=lambda x: x,
        )
    else :
        dataloader = []
    return dataloader