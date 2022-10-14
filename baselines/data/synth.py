import argparse
import torch
from torch import distributed as dist
import numpy as np
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from colossalai.nn.parallel.layers.cache_embedding import CachedEmbeddingBag
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
from torch.autograd.profiler import record_function
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
    if size == '52M':
        pass
    elif size == '4M':
        # 4210897
        CHOSEN_TABLES = [5, 8, 37, 54, 71, 72, 73, 74, 85, 86, 89, 95, 96, 97, 107, 131, 163, 185, 196, 204, 211]
        NUM_EMBEDDINGS_PER_FEATURE = '204008, 282795, 539726, 153492, 11644, 11645, 13858, 5632, 60121, \
            11711, 11645, 43335, 4843, 67919, 6539, 17076, 11579, 866124, 711855, 302001, 873349'
    elif size == '512M':
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
    elif size == '2G':
        # 2328942965
        CHOSEN_TABLES = [i for i in range(856)]
        NUM_EMBEDDINGS_PER_FEATURE = '8015999, 1, 9997799, 6138289, 21886, 204008, 220, 6148, 282795, 1316, 3639992, 1, 319, 1, 1, 3394206, 1, 1, 12203324, 1, 1, 1, 4091851, 1, 1, 1, 1, 11641, 4657566, 11645, 1815316, 6618925, 1, 1, 1146, 1, 1, 539726, 1, 1, 1, 4972387, 2169014, 1, 1, 3912105, 272, 3102763, 1, 1, 4230916, 5878, 1, 11645, 153492, 6618919, 1, 4868981, 1, 11709, 3985291, 1, 5956409, 1, 1, 1, 1, 1875019, 1, 381, 1, 11644, 11645, 13858, 5632, 1, 1, 6600102, 6618930, 1, 5412960, 371, 5272932, 2555079, 1, 60121, 11711, 2553205, 2647912, 11645, 1, 5798421, 350642, 1, 1, 43335, 4843, 67919, 1, 1, 3239310, 1, 1, 6855076, 1, 1, 1, 6539, 1, 1, 111, 5990989, 1, 6516585, 1, 68, 1, 5758479, 2448698, 1, 6618898, 2614073, 3309464, 1, 6107319, 1, 1, 1928793, 4535618, 1, 309, 17076, 4950876, 304795, 4970566, 11209763, 5585, 2207735, 6618921, 1941, 5659, 5690, 1029648, 5662, 4718, 6385214, 5641, 1150, 5653, 6618924, 1, 339750, 1, 6112009, 589094, 2844205, 1, 6618929, 1, 1, 5667, 5167062, 2542266, 11579, 6147171, 951851, 6448758, 5253, 826, 1, 1997119, 6363150, 6614703, 2199, 6461842, 913043, 1, 1, 1, 1, 1, 1283500, 1, 6316718, 11579, 866124, 3660331, 1, 4032709, 1, 1, 3232, 2065, 6584597, 1, 1, 711855, 5672538, 1, 248, 1, 1, 1, 1, 302001, 4006173, 1, 1, 19623, 1, 4673098, 873349, 8026000, 2323, 1680975, 1, 1, 5710807, 2999962, 5999910, 5925217, 4997507, 5999548, 2999938, 4999774, 5999707, 5999710, 5764956, 5999992, 1, 2999941, 5982534, 1, 5999927, 4978274, 5999983, 5999997, 5999912, 5908896, 5999955, 5999935, 5999836, 5999983, 1, 5999477, 5999805, 5998095, 1, 5989511, 1999998, 4999998, 6000000, 5999929, 1, 5999993, 1, 1, 5999885, 5999867, 5999929, 1, 1, 5999962, 1, 2999898, 5998777, 5999934, 1, 1, 5992524, 5999737, 2999538, 5999870, 5992524, 5999975, 1, 5710807, 1, 1, 4932124, 5918154, 1, 5997068, 5999982, 1, 5998551, 5999994, 5999870, 4999919, 5999944, 5999904, 4999740, 5922605, 5975615, 5999816, 998848, 5999926, 5999816, 4999991, 5999861, 1, 5999929, 5999885, 5999976, 1, 5999981, 5999901, 5999929, 1, 5999885, 5987787, 6000000, 5999929, 5998095, 1, 1, 3000000, 5999981, 2999993, 5999981, 5092210, 4999972, 5999976, 5998595, 1, 5999548, 1999882, 4998224, 5999929, 1, 5014074, 1, 1, 1, 1, 5999986, 5999978, 5999941, 5999816, 1, 5997022, 5999975, 1, 5999685, 5999981, 5999738, 5999380, 5966699, 5975615, 5908896, 5999996, 5999996, 5999983, 5734426, 5997022, 1, 5999975, 5999929, 5999996, 5999239, 5989271, 5999477, 5999981, 5999887, 1, 5999929, 5999506, 5999996, 5999548, 1, 5998472, 5922238, 5999975, 1, 1, 5987787, 1, 1, 2999964, 5999983, 1, 5999930, 5979767, 5999139, 5775261, 5999681, 4999929, 1, 5963607, 5999967, 2999835, 5997068, 5998595, 5999996, 5992524, 5999997, 5999932, 1, 5999878, 5999929, 5999857, 5999981, 1, 5999981, 1, 5999796, 5999995, 5994671, 1, 1, 5999329, 1, 1, 5997068, 5999981, 5973566, 5999407, 5966699, 5966699, 5734426, 5999975, 5999976, 1, 1, 1, 2982258, 5999816, 5999929, 5999981, 1, 5999974, 5998583, 5966699, 5999870, 5999828, 5997903, 5999854, 5999685, 1, 1, 1, 5999954, 5999981, 1, 1, 5999967, 5999681, 5999932, 1, 5999857, 5971899, 5999972, 5999932, 1, 5999979, 1, 5998507, 5999927, 5999981, 5998595, 5999975, 1, 5949097, 5999239, 5999821, 1, 1, 5922605, 1, 5998473, 5999878, 5992217, 5999600, 1, 1, 5965155, 5999932, 5999971, 5922605, 5999854, 1, 5999963, 1, 5998777, 5999816, 1, 5999594, 1, 5733183, 5999396, 1, 1, 1, 5999440, 1, 5871486, 5999975, 1, 5999911, 5963607, 5997079, 1, 5999772, 1, 5999857, 5999863, 5999477, 5966699, 5999975, 5999996, 5999976, 1, 5999981, 6, 135, 422, 509, 922, 13, 16, 2, 1000, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1245, 5, 3136, 2704, 4999983, 4999983, 4999937, 4000, 1, 1, 1, 1, 1, 1, 1, 5, 6, 5, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 8, 2, 2, 1, 1, 2, 2, 8, 999908, 1443441, 21812, 21656, 14, 14, 1, 61, 2043, 2, 137, 13, 5, 10, 10, 10, 10, 9, 10, 12, 11, 10, 11, 11, 16, 10, 10, 11, 16, 1, 1, 1, 12543437, 1, 4999993, 4999986, 3999995, 4999863, 12543669, 1, 4999998, 12543650, 4999985, 12540349, 1, 1, 12532399, 4999998, 12540535, 397606, 1415772, 12270355, 9765324, 1, 1, 10287986, 1, 10794735, 10498728, 10965644, 4667, 43200, 43200, 43200, 43200, 629, 226, 215, 24, 4999983, 1028203, 20, 10, 1, 1, 1, 1, 12542, 2, 85, 2, 2, 2, 8, 2, 4, 2, 2, 2, 2, 2, 4, 6, 2, 2, 2, 8, 4, 9810, 2, 5884133, 1, 1, 2, 4, 2, 2, 1, 2, 4, 2, 18, 18, 2, 4, 6137684, 3309463, 1, 5607839, 1, 1, 1, 6567668, 1, 1, 4093617, 1, 4390473, 6305161, 1, 1, 1, 3779483, 1, 5303395, 1, 1, 6618931, 1, 1, 3640447, 3102628, 2542714, 269, 1, 1, 612, 6107445, 3978163, 6607315, 4868894, 1, 3983088, 5419139, 1, 271, 3911645, 2553341, 983482, 272, 1, 1600937, 1, 1266741, 1520037, 3704018, 1, 1, 3345638, 6618817, 3117219, 1, 1, 1877662, 1876652, 3309463, 4378405, 847629, 1661, 621, 624, 3667331, 1, 269, 2614200, 1, 1, 1, 6618759, 204, 1, 6618922, 1, 5998824, 5999974, 4977045, 5999994, 5999995, 5999993, 5999981, 5999957, 5999908, 5984869, 5999994, 1, 5999548, 1, 1, 5999831, 5999978, 5999396, 5999908, 5999953, 1, 1, 1, 5999750, 5999958, 5999477, 1, 5999981, 5999548, 5999953, 1, 5999548, 1, 1, 5999986, 5999975, 1, 5999908, 1, 1, 5999975, 1, 5999548, 5998836, 5999477, 5999737, 5999708, 5999737, 5999783, 1, 1, 5999901, 5999708, 5999711, 5999967, 5999548, 5999548, 1, 1, 5999783, 1, 5999694, 5999520, 5999975, 1, 1, 1, 5999477, 1, 5999975, 5991327, 5975615, 1, 4494193, 5999918, 1, 5999725, 1, 5999995, 1, 1, 5999477, 4998981, 5999975, 5999329, 5976924, 1, 2, 4999540, 5999138, 3994232, 1'
    else:
        raise NotImplementedError()
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
                # slice_ = slice(row_idx, row_idx + rows_to_get)
                with record_function("## load_batch ##"):
                    sparse_indices, sparse_lengths = self._load_slice_batch(self.indices_per_table_per_file[file_idx],
                                                                             self.offsets_per_table_per_file[file_idx],
                                                                             self.lengths_per_table_per_file[file_idx],
                                                                            row_idx,
                                                                            rows_to_get
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
    
    def _load_slice_batch(self, indices_per_table, offsets_per_table, lengths_per_table, row_start, row_length):
        chosen_indices_list = []
        chosen_lengths_list = []
        for indices, offsets, lengths in zip(indices_per_table, offsets_per_table, lengths_per_table):
            chosen_lengths_list.append(lengths.narrow(0, row_start, row_length))
            start = offsets[row_start]
            end = offsets[row_start + row_length]
            chosen_indices_list.append(indices.narrow(0, start, end - start))
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
    files = filter(lambda s: "fbgemm_t856_bs65536_" in s, files)
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