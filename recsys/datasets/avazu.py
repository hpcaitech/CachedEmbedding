import torch.utils.data.datapipes as dp
from torch.utils.data import IterDataPipe, IterableDataset
from torchrec.datasets.utils import LoadFiles, ReadLinesFromCSV, PATH_MANAGER_KEY
from torchrec.datasets.criteo import BinaryCriteoUtils

CAT_FEATURE_COUNT = 21
DAYS = 10
DEFAULT_LABEL_NAME = "label"
DEFAULT_CAT_NAMES = [
    'C1', 'banner_pos', 'site_id', 'site_domain', 'site_category', 'app_id', 'app_domain', 'app_category', 'device_id',
    'device_ip', 'device_model', 'device_type', 'device_conn_type', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20',
    'C21'
]
NUM_EMBEDDINGS_PER_FEATURE = '7,7,4737,7745,26,8552,559,36,2686408,6729486,8251,5,4,2626,8,9,435,4,68,172,60'
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
                 sparse_paths,
                 label_paths,
                 batch_size,
                 rank,
                 world_size,
                 shuffle_batches=False,
                 mmap_mode=False,
                 hashes=None,
                 path_manager_key=PATH_MANAGER_KEY):
        self.sparse_paths = sparse_paths
        self.label_paths = label_paths
        self.batch_size = batch_size
        self.rank = rank
        self.world_size = world_size
        self.shuffle_batches = shuffle_batches
        self.mmap_mode = mmap_mode
        self.hashes = hashes
        self.path_manager_key = path_manager_key

        self._load_data()

    def _load_data(self):
        file_idx_to_row_range = BinaryCriteoUtils.get_file_idx_to_row_range(
            lengths=[BinaryCriteoUtils.get_shape_from_npy(self.sparse_paths, self.path_manager_key)[0]],
            rank=self.rank,
            world_size=self.world_size)
        self.sparse_arrs, self.labels_arrs = [], []
        for arrs, paths in zip([self.sparse_arrs, self.labels_arrs], [self.sparse_paths, self.label_paths]):
            for idx, (range_left, range_right) in file_idx_to_row_range.items():
                arrs.append(
                    BinaryCriteoUtils.load_npy_range(paths[idx],
                                                     range_left,
                                                     range_right - range_left + 1,
                                                     path_manager_key=self.path_manager_key,
                                                     mmap_mode=self.mmap_mode))
        if not self.mmap_mode and self.hashes is not None:
            for sparse_arr in self.sparse_arrs:
                sparse_arr %= self.hashes
                sparse_arr += self.sparse_offsets

    def __iter__(self):
        ...
