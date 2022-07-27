# This script preprocesses Criteo dataset tsv files to binary (npy) files.

# In order to exploit InMemoryBinaryCriteoIterDataPipe to accelerate the loading of data,
# this file is modified from torchrec/datasets/scripts/npy_preproc_criteo.py and
# torchrec.datasets.criteo
#
# Usage:
#       python npy_preproc_criteo.py --input_dir /where/criteo_kaggle/train.txt --output_dir /where/to/save/.npy
#
# You may need additional modifications for the file name,
# and you can evenly split the whole dataset into 7 days by split_criteo_kaggle.py

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
from torchrec.datasets.criteo import CriteoIterDataPipe, INT_FEATURE_COUNT, CAT_FEATURE_COUNT
from torchrec.datasets.utils import PATH_MANAGER_KEY, safe_cast
from iopath.common.file_io import PathManagerFactory


def tsv_to_npys(
    in_file: str,
    out_dense_file: str,
    out_sparse_file: str,
    out_labels_file: str,
    path_manager_key: str = PATH_MANAGER_KEY,
):
    """
    For criteo kaggle
    """

    def row_mapper(row: List[str]) -> Tuple[List[int], List[int], int]:
        label = safe_cast(row[0], int, 0)
        dense = [safe_cast(row[i], int, 0) for i in range(1, 1 + INT_FEATURE_COUNT)]
        sparse = [
            int(safe_cast(row[i], str, "0") or "0", 16)
            for i in range(1 + INT_FEATURE_COUNT, 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT)
        ]
        return dense, sparse, label    # pyre-ignore[7]

    dense, sparse, labels = [], [], []
    for (row_dense, row_sparse, row_label) in CriteoIterDataPipe([in_file], row_mapper=row_mapper):
        dense.append(row_dense)
        sparse.append(row_sparse)
        labels.append(row_label)

    dense_np = np.array(dense, dtype=np.int32)
    del dense
    sparse_np = np.array(sparse, dtype=np.int32)
    del sparse
    labels_np = np.array(labels, dtype=np.int32)
    del labels

    # Why log +3?
    dense_np -= (dense_np.min() - 2)
    dense_np = np.log(dense_np, dtype=np.float32)

    labels_np = labels_np.reshape((-1, 1))
    path_manager = PathManagerFactory().get(path_manager_key)
    for (fname, arr) in [
        (out_dense_file, dense_np),
        (out_sparse_file, sparse_np),
        (out_labels_file, labels_np),
    ]:
        with path_manager.open(fname, "wb") as fout:
            np.save(fout, arr)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Criteo tsv -> npy preprocessing script.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing Criteo tsv files. Files in the directory "
        "should be named day_{0-23}.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to store npy files.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    """
    This function preprocesses the raw Criteo tsvs into the format (npy binary)
    expected by InMemoryBinaryCriteoIterDataPipe.

    Args:
        argv (List[str]): Command line args.

    Returns:
        None.
    """

    args = parse_args(argv)
    input_dir = args.input_dir
    output_dir = args.output_dir

    for f in os.listdir(input_dir):
        in_file_path = os.path.join(input_dir, f)
        dense_out_file_path = os.path.join(output_dir, f + "_dense.npy")
        sparse_out_file_path = os.path.join(output_dir, f + "_sparse.npy")
        labels_out_file_path = os.path.join(output_dir, f + "_labels.npy")
        print(f"Processing {in_file_path}. Outputs will be saved to {dense_out_file_path}"
              f", {sparse_out_file_path}, and {labels_out_file_path}...")
        tsv_to_npys(
            in_file_path,
            dense_out_file_path,
            sparse_out_file_path,
            labels_out_file_path,
        )
        print(f"Done processing {in_file_path}.")


if __name__ == "__main__":
    main(sys.argv[1:])
