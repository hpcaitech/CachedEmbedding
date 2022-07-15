from typing import Tuple, List, Optional, Iterator
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import math
import struct

import torch
import torch.utils.data
from torch.utils.data.dataset import IterableDataset
import numpy as np
from tqdm import tqdm
import lmdb

INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]


@dataclass
class Batch:
    dense_features: torch.Tensor
    sparse_features: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(
            dense_features=self.dense_features.to(
                device=device, non_blocking=non_blocking
            ),
            sparse_features=self.sparse_features.to(
                device=device, non_blocking=non_blocking
            ),
            labels=self.labels.to(device=device, non_blocking=non_blocking),
        )

    def record_stream(self, stream: torch.cuda.streams.Stream) -> None:
        self.dense_features.record_stream(stream)
        self.sparse_features.record_stream(stream)
        self.labels.record_stream(stream)

    def pin_memory(self) -> "Batch":
        return Batch(
            dense_features=self.dense_features.pin_memory(),
            sparse_features=self.sparse_features.pin_memory(),
            labels=self.labels.pin_memory(),
        )


class _RandomRecBatch:
    generator: Optional[torch.Generator]

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

        if manual_seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(manual_seed)
        else:
            self.generator = None

        self._generated_batches: List[Tuple] = [
            self._generate_batch()
        ] * num_generated_batches
        self.batch_index = 0

    def __iter__(self) -> "_RandomRecBatch":
        self.batch_index = 0
        return self

    def __next__(self) -> Batch:
        if self.batch_index == self.num_batches:
            raise StopIteration
        if self.num_generated_batches >= 0:
            batch = self._generated_batches[
                self.batch_index % len(self._generated_batches)
            ]
        else:
            batch = self._generate_batch()
        self.batch_index += 1
        return batch

    def _generate_batch(self) -> Batch:

        values = []
        for key_idx, _ in enumerate(self.keys):
            hash_size = self.hash_sizes[key_idx]
            num_ids_in_batch = self.ids_per_features[key_idx]

            values.append(
                # pyre-ignore
                torch.randint(
                    high=hash_size,
                    size=(num_ids_in_batch * self.batch_size,),
                    generator=self.generator,
                )
            )
            
        sparse_features = torch.cat([y.unsqueeze(1) for y in values], dim=1)
        
        dense_features = torch.randn(
            self.batch_size,
            self.num_dense,
            generator=self.generator,
        )

        labels = torch.randint(
            low=0,
            high=2,
            size=(self.batch_size,),
            generator=self.generator,
        )

        batch = Batch(
            sparse_features=sparse_features,
            dense_features=dense_features,
            labels=labels.float(),
        )
        return batch


class RandomCriteoDataset(IterableDataset[Batch]):

    def __init__(
        self,
        keys: List[str],
        batch_size: int,
        hash_size: Optional[int] = 100,
        hash_sizes: Optional[List[int]] = None,
        ids_per_feature: Optional[int] = 2,
        ids_per_features: Optional[List[int]] = None,
        num_dense: int = 50,
        manual_seed: Optional[int] = None,
        num_batches: Optional[int] = None,
        num_generated_batches: int = 10,
    ) -> None:
        super().__init__()

        if hash_sizes is None:
            hash_size = hash_size or 100
            hash_sizes = [hash_size] * len(keys)

        assert hash_sizes is not None
        assert len(hash_sizes) == len(
            keys
        ), "length of hash_sizes must be equal to the number of keys"

        if ids_per_features is None:
            ids_per_feature = ids_per_feature or 2
            ids_per_features = [ids_per_feature] * len(keys)

        assert ids_per_features is not None
        assert len(ids_per_features) == len(
            keys
        ), "length of ids_per_features must be equal to the number of keys"

        self.batch_generator = _RandomRecBatch(
            keys=keys,
            batch_size=batch_size,
            hash_sizes=hash_sizes,
            ids_per_features=ids_per_features,
            num_dense=num_dense,
            manual_seed=manual_seed,
            num_batches=num_batches,
            num_generated_batches=num_generated_batches,
        )

    def __iter__(self) -> Iterator[Batch]:
        return iter(self.batch_generator)


class CriteoDataset(IterableDataset[Batch]):
    """
    Criteo Display Advertising Challenge Dataset
    Data prepration:
        * Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
        * Discretize numerical values by log2 transformation which is proposed by the winner of Criteo Competition
    :param dataset_path: criteo train.txt path.
    :param cache_path: lmdb cache path.
    :param rebuild_cache: If True, lmdb cache is refreshed.
    :param min_threshold: infrequent feature threshold.
    Reference:
        https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset
        https://www.csie.ntu.edu.tw/~r01922136/kaggle-2014-criteo.pdf
    """

    def __init__(self, args, rebuild_cache=False, min_threshold=10, mode='train'):
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        self.batch_size = args.batch_size
        self.mode = mode
        if not Path(args.cache_path).exists() or not rebuild_cache:
            print('Random dataset generated')
            self.env = None
            setattr(self, f'{mode}_dataset', RandomCriteoDataset(
                    keys=DEFAULT_CAT_NAMES,
                    batch_size=args.batch_size,
                    hash_size=args.num_embeddings,
                    hash_sizes=args.num_embeddings_per_feature if hasattr(args, "num_embeddings_per_feature") else None,
                    manual_seed=args.seed if hasattr(args, "seed") else None,
                    ids_per_feature=1,
                    num_dense=len(DEFAULT_INT_NAMES),
                    num_batches=getattr(args, f"limit_{mode}_batches"),))
        else:
            self.env = lmdb.open(args.cache_path, create=False, lock=False, readonly=True)
            with self.env.begin(write=False) as txn:
                self.total_len = txn.stat()['entries'] - 1
                self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)
                self.length = getattr(args, f"limit_{mode}_batches") * self.batch_size
                if mode == 'train':
                    self.curr_index = 0
                elif mode == 'val':
                    self.curr_index = getattr(args, f"limit_train_batches") * self.batch_size
                else:
                    self.curr_index = (getattr(args, f"limit_val_batches") + \
                        getattr(args, f"limit_train_batches")) * self.batch_size
        
    def __next__(self):
        dense_features = []
        sparse_features = []
        labels = []
        for index in range(self.curr_index, self.curr_index+self.batch_size):
            dense_feature, sparse_feature, label = self.__getitem__(index)
            dense_features.append(dense_feature)
            sparse_features.append(sparse_feature)
            labels.append(label)

        self.curr_index = self.curr_index + self.batch_size
        
        if self.curr_index > self.length:
            raise StopIteration
            
        return Batch(dense_features=torch.cat([x.unsqueeze(0) for x in dense_features],dim=0),
                    sparse_features=torch.cat([x.unsqueeze(0) for x in sparse_features],dim=0), 
                    labels=torch.tensor(labels))
        
    def __iter__(self):
        if self.env is not None:
            self.curr_index = 0
            return self
        else:
            return iter(getattr(self, f'{self.mode}_dataset'))

    def __getitem__(self, index):
        if self.env is not None:
            with self.env.begin(write=False) as txn:
                np_array = np.frombuffer(
                    txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
            return (torch.from_numpy(np_array[1:self.NUM_INT_FEATS+1]).to(dtype=torch.float),\
                        torch.from_numpy(np_array[self.NUM_INT_FEATS+1:]),\
                        torch.tensor(np_array[0]))
        else:
            raise NotImplementedError()

    def __len__(self):
        if self.env is not None:
            return self.length
        else:
            raise NotImplementedError()

    ## Deprecated ##
    def _build_cache(self, path, cache_path):
        feat_mapper, defaults = self._get_feat_mapper(path)
        with lmdb.open(cache_path, map_size=int(1e11)) as env:
            field_dims = np.zeros(self.NUM_FEATS, dtype=np.uint32)
            for i, fm in feat_mapper.items():
                field_dims[i - 1] = len(fm) + 1
            with env.begin(write=True) as txn:
                txn.put(b'field_dims', field_dims.tobytes())
            for buffer in self._yield_buffer(path, feat_mapper, defaults):
                with env.begin(write=True) as txn:
                    for key, value in buffer:
                        txn.put(key, value)

    def _get_feat_mapper(self, path):
        feat_cnts = defaultdict(lambda: defaultdict(int))
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                for i in range(1, self.NUM_INT_FEATS + 1):
                    feat_cnts[i][convert_numeric_feature(values[i])] += 1
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def _yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create criteo dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[0])
                for i in range(1, self.NUM_INT_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(convert_numeric_feature(values[i]), defaults[i])
                for i in range(self.NUM_INT_FEATS + 1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer


def convert_numeric_feature(val: str):
    if val == '':
        return 'NULL'
    v = int(val)
    if v > 2:
        return str(int(math.log(v) ** 2))
    else:
        return str(v - 2)

