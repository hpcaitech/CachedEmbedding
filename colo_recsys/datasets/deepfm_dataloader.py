import math
import shutil
import struct
from collections import defaultdict
from pathlib import Path

import lmdb
import torch
import torch.utils.data
import numpy as np
import pandas as pd
from tqdm import tqdm

class MovieLens20MDataset(torch.utils.data.Dataset):
    """
    MovieLens 1B Dataset
    Data preparation
        treat samples with a rating less than 3 as negative samples
    :param dataset_path: MovieLens dataset path
    Reference:
        https://grouplens.org/datasets/movielens
    """

    def __init__(self, dataset_path, sep=',', engine='c', header='infer'):
        data = pd.read_csv(dataset_path, sep=sep, engine=engine, header=header).to_numpy()[:, :3]
        self.items = data[:, :2].astype(np.int) - 1  # -1 because ID begins from 1
        self.targets = self._preprocess_target(data[:, 2]).astype(np.float32)
        self.field_dims = np.max(self.items, axis=0) + 1
        self.user_field_idx = np.array((0, ), dtype=np.long)
        self.item_field_idx = np.array((1,), dtype=np.long)

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def _preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target


class AvazuDataset(torch.utils.data.Dataset):
    """
    Avazu Click-Through Rate Prediction Dataset
    Dataset preparation
        Remove the infrequent features (appearing in less than threshold instances) and treat them as a single feature
    :param dataset_path: avazu train path
    :param cache_path: lmdb cache path
    :param rebuild_cache: If True, lmdb cache is refreshed
    :param min_threshold: infrequent feature threshold
    Reference
        https://www.kaggle.com/c/avazu-ctr-prediction
    """

    def __init__(self, dataset_path=None, cache_path='.avazu', rebuild_cache=False, min_threshold=4):
        self.NUM_FEATS = 22
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self.__build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

    def __build_cache(self, path, cache_path):
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
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: counting features')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i + 1]] += 1
        feat_mapper = {i: {feat for feat, c in cnt.items() if c >= self.min_threshold} for i, cnt in feat_cnts.items()}
        feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)} for i, cnt in feat_mapper.items()}
        defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        return feat_mapper, defaults

    def _yield_buffer(self, path, feat_mapper, defaults, buffer_size=int(1e5)):
        item_idx = 0
        buffer = list()
        with open(path) as f:
            f.readline()
            pbar = tqdm(f, mininterval=1, smoothing=0.1)
            pbar.set_description('Create avazu dataset cache: setup lmdb')
            for line in pbar:
                values = line.rstrip('\n').split(',')
                if len(values) != self.NUM_FEATS + 2:
                    continue
                np_array = np.zeros(self.NUM_FEATS + 1, dtype=np.uint32)
                np_array[0] = int(values[1])
                for i in range(1, self.NUM_FEATS + 1):
                    np_array[i] = feat_mapper[i].get(values[i+1], defaults[i])
                buffer.append((struct.pack('>I', item_idx), np_array.tobytes()))
                item_idx += 1
                if item_idx % buffer_size == 0:
                    yield buffer
                    buffer.clear()
            yield buffer

class CriteoDataset(torch.utils.data.Dataset):
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

    def __init__(self, dataset_path=None, cache_path='.criteo', rebuild_cache=False, min_threshold=10):
        self.NUM_FEATS = 39
        self.NUM_INT_FEATS = 13
        self.min_threshold = min_threshold
        if rebuild_cache or not Path(cache_path).exists():
            shutil.rmtree(cache_path, ignore_errors=True)
            if dataset_path is None:
                raise ValueError('create cache: failed: dataset_path is None')
            self._build_cache(dataset_path, cache_path)
        self.env = lmdb.open(cache_path, create=False, lock=False, readonly=True)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries'] - 1
            self.field_dims = np.frombuffer(txn.get(b'field_dims'), dtype=np.uint32)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            np_array = np.frombuffer(
                txn.get(struct.pack('>I', index)), dtype=np.uint32).astype(dtype=np.long)
        return np_array[1:], np_array[0]

    def __len__(self):
        return self.length

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

