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



# 52667139 as default
CHOSEN_TABLES = [0, 2, 3, 4, 5, 7, 8, 9, 10, 12, 15, 18, 22, 27, 28]
NUM_EMBEDDINGS_PER_FEATURE = '8015999, 9997799, 6138289, 21886, 204008, 6148, 282795, \
    1316, 3639992, 319, 3394206, 12203324, 4091851, 11641, 4657566'

CAT_FEATURE_COUNT = len(CHOSEN_TABLES)
INT_FEATURE_COUNT = 1
DEFAULT_LABEL_NAME = "click"
DEFAULT_INT_NAMES = ['rand_dense']
BATCH_SIZE = 65536 # batch_size of one file
DEFAULT_CAT_NAMES = ["cat_{}".format(i) for i in range(len(CHOSEN_TABLES))]

def choose_data_size(size: str):
    global CHOSEN_TABLES
    global NUM_EMBEDDINGS_PER_FEATURE
    global CAT_FEATURE_COUNT
    global DEFAULT_CAT_NAMES
    if size == 'middle':
        pass
    elif size == 'small':
        # 4210897
        CHOSEN_TABLES = [5, 8, 37, 54, 71, 72, 73, 74, 85, 86, 89, 95, 96, 97, 107, 131, 163, 185, 196, 204, 211]
        NUM_EMBEDDINGS_PER_FEATURE = '204008, 282795, 539726, 153492, 11644, 11645, 13858, 5632, 60121, \
            11711, 11645, 43335, 4843, 67919, 6539, 17076, 11579, 866124, 711855, 302001, 873349'
    elif size == 'big':
        # 512196316
        CHOSEN_TABLES = [301, 302, 303, 305, 306, 307, 309, 310, 311, 312, 313, 316, 317, 318, 319, 320, 321, 322, 323,
                         325, 326, 327, 328, 330, 335, 336, 337, 338, 340, 341, 343, 344, 345, 346, 347, 348, 349, 350,
                         351, 352, 353, 354, 356, 357, 358, 359, 360, 361, 362, 363, 365, 366, 367, 368, 370, 371, 372,
                         375, 378, 379, 381, 382, 383, 384, 385, 386, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398,
                         399, 400, 401, 403, 405, 406, 407, 410, 413, 414, 415, 416, 417]
        NUM_EMBEDDINGS_PER_FEATURE = '5999929, 5999885, 5999976, 5999981, 5999901, 5999929, 5999885, 5987787, 6000000, 5999929, \
            5998095, 3000000, 5999981, 2999993, 5999981, 5092210, 4999972, 5999976, 5998595, 5999548, 1999882, 4998224, 5999929, \
                5014074, 5999986, 5999978, 5999941, 5999816, 5997022, 5999975, 5999685, 5999981, 5999738, 5999380, 5966699, 5975615, \
                    5908896, 5999996, 5999996, 5999983, 5734426, 5997022, 5999975, 5999929, 5999996, 5999239, 5989271, 5999477, 5999981, \
                        5999887, 5999929, 5999506, 5999996, 5999548, 5998472, 5922238, 5999975, 5987787, 2999964, 5999983, 5999930, 5979767, \
                            5999139, 5775261, 5999681, 4999929, 5963607, 5999967, 2999835, 5997068, 5998595, 5999996, 5992524, 5999997, 5999932, \
                                5999878, 5999929, 5999857, 5999981, 5999981, 5999796, 5999995, 5994671, 5999329, 5997068, 5999981, 5973566, 5999407, 5966699'
    else:
        pass # middle
    CAT_FEATURE_COUNT = len(CHOSEN_TABLES)
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
        
        self._buffer = None
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
        # self._buffer structure:
        '''
        self._buffer[0]: List of sparse_indices per table
        self._buffer[1]: List of sparse_lengths per table
        '''
        def append_to_buffer(sparse_indices: List[torch.Tensor], sparse_lengths: List[torch.Tensor]):
            if self._buffer is None:
                self._buffer = [sparse_indices, sparse_lengths]
            else:
                for tb_idx, (sparse_indices_table, sparse_lengths_table) in enumerate(zip(sparse_indices, sparse_lengths)):
                    self._buffer[0][tb_idx] = torch.cat((self._buffer[0][tb_idx], sparse_indices_table))
                    self._buffer[1][tb_idx] = torch.cat((self._buffer[1][tb_idx], sparse_lengths_table))
                    
        file_idx = 0
        row_idx = 0
        batch_idx = 0
        while batch_idx < self.num_batches:
            buffer_row_count = 0 if self._buffer is None else self._buffer[1][0].shape[0]
            if buffer_row_count == self.batch_size:
                yield self._make_batch(*self._buffer)
                batch_idx += 1
                self._buffer = None
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