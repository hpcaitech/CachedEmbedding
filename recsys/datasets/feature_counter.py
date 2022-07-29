import abc
import numpy as np
import os
from . import criteo


def get_criteo_id_freq_map(path):
    files = os.listdir(path)
    sparse_files = list(filter(lambda s: 'sparse' in s, files))
    sparse_files = [os.path.join(path, _f) for _f in sparse_files]

    file_processor = CriteoSparseProcessor(list(map(int, criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(','))))
    feature_count = GlobalFeatureCounter(sparse_files, file_processor)

    return feature_count.id_freq_map


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
