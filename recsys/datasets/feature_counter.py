import abc
import itertools
from tqdm import tqdm

import numpy as np
from contexttimer import Timer
import torch
from torch.utils.data import DataLoader
try:
    # pyre-ignore[21]
    import nvtabular as nvt
    from nvtabular.loader.torch import TorchAsyncItr
except ImportError:
    print("Unable to import NVTabular, which indicates that you cannot load criteo 1TB dataset with our solution")

from .criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
from .utils import KJTTransform


class CriteoSparseProcessor:

    def __init__(self, hash_sizes):
        self.hash_sizes = np.array(hash_sizes).reshape(1, -1)
        self.offsets = np.array([0, *np.cumsum(hash_sizes)[:-1]]).reshape(1, -1)

    def __call__(self, _f):
        arr = np.load(_f)
        arr %= self.hash_sizes
        arr += self.offsets
        flattened = arr.reshape(-1)
        bins = np.bincount(flattened, minlength=self.hash_sizes.sum())
        return bins


class BaseFeatureCounter(abc.ABC):

    def __init__(self, datafiles):
        self.datafiles = datafiles
        self._id_freq_map = None
        self._collect_statistics()

    @abc.abstractmethod
    def _collect_statistics(self):
        pass

    @property
    def id_freq_map(self):
        return self._id_freq_map


class GlobalFeatureCounter(BaseFeatureCounter):
    """
    compute the global statistics of the whole training set
    """

    def __init__(self, datafiles, file_callback):
        self.file_processor = file_callback

        super(GlobalFeatureCounter, self).__init__(datafiles)

    def _collect_statistics(self):
        for _f in self.datafiles:
            if self._id_freq_map is None:
                self._id_freq_map = self.file_processor(_f)
            else:
                self._id_freq_map += self.file_processor(_f)


class NVTabularFeatureCounter:

    def __init__(self, datafiles, hashes, batch_size, sample_fraction=0.05):
        self.datafiles = datafiles
        self._id_freq_map = torch.zeros(sum(hashes), dtype=torch.long)
        self.batch_size = batch_size
        self.pre_ones = torch.ones(batch_size * len(DEFAULT_CAT_NAMES), dtype=torch.long)
        self.sample_fraction = sample_fraction
        self._collect_statistics()

    def _collect_statistics(self):
        data_files = sorted(self.datafiles[:int(np.ceil(len(self.datafiles) * self.sample_fraction))])
        nv_iter = TorchAsyncItr(
            nvt.Dataset(data_files, engine="parquet", part_mem_fraction=0.02),
            batch_size=self.batch_size,
            cats=DEFAULT_CAT_NAMES,
            conts=DEFAULT_INT_NAMES,
            labels=["label"],
            global_rank=0,
            global_size=1,
            drop_last=False,
            device='cpu',
        )

        dataloader = DataLoader(nv_iter,
                                batch_size=None,
                                pin_memory=False,
                                collate_fn=KJTTransform(nv_iter).transform,
                                num_workers=0)
        data_iter = iter(dataloader)
        with Timer() as timer:
            for it in tqdm(itertools.count()):
                try:
                    sparse = next(data_iter).sparse_features.values()
                    ones = self.pre_ones.narrow(0, start=0, length=sparse.shape[0])
                    self._id_freq_map.index_add_(dim=0, index=sparse, source=ones)
                except StopIteration:
                    break
        print(f"collect statistics over files: {data_files} num batch: {len(dataloader)}, batch size: {self.batch_size}"
              f", average time cost: {len(dataloader) / timer.elapsed:.2f} batch/s")

    @property
    def id_freq_map(self):
        return self._id_freq_map
