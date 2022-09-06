import abc
import random
from tqdm import tqdm

import numpy as np
from .criteo import DEFAULT_CAT_NAMES
from petastorm import make_batch_reader
from pyarrow.parquet import ParquetDataset


class GlobalFeatureCounter:
    """
    compute the global statistics of the whole training set
    """

    def __init__(self, datafiles, hash_sizes):
        self.datafiles = datafiles
        self.hash_sizes = np.array(hash_sizes).reshape(1, -1)
        self.offsets = np.array([0, *np.cumsum(hash_sizes)[:-1]]).reshape(1, -1)

    def compute(self):
        id_freq_map = np.zeros(self.hash_sizes.sum(), dtype=np.int64)
        for _f in self.datafiles:
            arr = np.load(_f)
            arr %= self.hash_sizes
            arr += self.offsets
            flattened = arr.reshape(-1)
            id_freq_map += np.bincount(flattened, minlength=self.hash_sizes.sum())
        return id_freq_map

class PetastormCounter:

    def __init__(self, datafiles, hash_sizes, subsample_fraction=0.2, seed=1024):
        self.datafiles = datafiles
        self.total_features = sum(hash_sizes)

        self.offsets = np.array([0, *np.cumsum(hash_sizes)[:-1]]).reshape(1, -1)
        self.subsample_fraction = subsample_fraction
        self.seed = seed

    def compute(self):
        _id_freq_map = np.zeros(self.total_features, dtype=np.int64)

        files = self.datafiles
        random.seed(self.seed)
        random.shuffle(files)
        if 0. < self.subsample_fraction < 1.:
            files = files[:int(np.ceil(len(files) * self.subsample_fraction))]

        dataset = ParquetDataset(files, use_legacy_dataset=False)
        with make_batch_reader(list(map(lambda x: "file://" + x, dataset.files)), num_epochs=1) as reader:
            for batch in tqdm(reader,
                              ncols=0,
                              desc="Collecting id-freq map",
                              total=sum([fragment.metadata.num_row_groups for fragment in dataset.fragments])):
                sparse = np.concatenate([getattr(batch, col_name).reshape(-1, 1) for col_name in DEFAULT_CAT_NAMES],
                                        axis=1)
                sparse = (sparse + self.offsets).reshape(-1)
                _id_freq_map += np.bincount(sparse, minlength=self.total_features)
        return _id_freq_map
