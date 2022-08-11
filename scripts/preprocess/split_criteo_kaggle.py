# This script adapts criteo kaggle dataset into 7 days' data
#
# Please alter the arguments of the two functions here.
#
# Usage:
#       python split_criteo_kaggle.py

import numpy as np
import os

from torchrec.datasets.criteo import BinaryCriteoUtils, CAT_FEATURE_COUNT


def main(data_dir, output_dir, days=7):
    STAGES = ("labels", "dense", "sparse")
    files = [os.path.join(data_dir, f"train.txt_{split}.npy") for split in STAGES]
    total_rows = BinaryCriteoUtils.get_shape_from_npy(files[0])[0]

    indices = list(range(0, total_rows, total_rows // days))
    ranges = []
    for i in range(len(indices) - 1):
        left_idx = indices[i]
        right_idx = indices[i + 1] if i < len(indices) - 2 else total_rows
        ranges.append((left_idx, right_idx))

    for _s, _f in zip(STAGES, files):
        for day, (left_idx, right_idx) in enumerate(ranges):
            chunk = BinaryCriteoUtils.load_npy_range(_f, left_idx, right_idx - left_idx)
            output_fname = f"day_{day}_{_s}.npy"
            np.save(os.path.join(output_dir, output_fname), chunk)


def get_num_embeddings_per_feature(path_to_sparse):
    sparse = np.load(path_to_sparse)
    assert sparse.shape[1] == CAT_FEATURE_COUNT

    nums = []
    for i in range(CAT_FEATURE_COUNT):
        nums.append(len(np.unique(sparse[:, i])))
    print(','.join(map(lambda x: str(x), nums)))


if __name__ == '__main__':
    main('/data/scratch/RecSys/criteo_kaggle_npy', 'criteo_kaggle')
    get_num_embeddings_per_feature('/data/scratch/RecSys/criteo_kaggle_npy/train.txt_sparse.npy')
