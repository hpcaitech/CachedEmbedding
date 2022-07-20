import os
import numpy as np
import pytest

from recsys.datasets.criteo import KAGGLE_NUM_EMBEDDINGS_PER_FEATURE
from recsys.datasets.feature_counter import CriteoSparseProcessor, GlobalFeatureCounter


@pytest.mark.skip("Only for local env in which the dataset_dir exists")
def test_feature_counter(dataset_dir):
    files = os.listdir(dataset_dir)
    sparse_files = list(filter(lambda s: 'sparse' in s, files))
    sparse_files = [os.path.join(dataset_dir, _f) for _f in sparse_files]

    file_processor = CriteoSparseProcessor(list(map(int, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(','))))
    feature_count = GlobalFeatureCounter(sparse_files, file_processor)

    sparse_data = []
    for each in sparse_files:
        sparse_data.append(np.load(each))
    sparse_data = np.vstack(sparse_data)
    sparse_data %= file_processor.hash_sizes
    sparse_data += file_processor.offsets
    ref = np.bincount(sparse_data.reshape(-1), minlength=sum(map(int, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(','))))

    np.allclose(feature_count.id_freq_map, ref)


if __name__ == "__main__":
    test_feature_counter("../criteo_kaggle")
