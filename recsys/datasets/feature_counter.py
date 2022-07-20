import abc
import numpy as np


class CriteoSparseProcessor:

    def __init__(self, hash_sizes):
        self.hash_sizes = np.array(hash_sizes).reshape(1, -1)
        self.offsets = np.array([0, *np.cumsum(hash_sizes)[:-1]]).reshape(1, -1)

    def __call__(self, _f):
        arr = np.load(_f)
        print(arr.shape, arr.dtype)
        print(self.offsets.shape, self.offsets.dtype)
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


class SubsetFeatureCounter(BaseFeatureCounter):
    """
    compute frequency-related statistics on the subsampled dataset
    """

    def _collect_statistics(self):
        pass
