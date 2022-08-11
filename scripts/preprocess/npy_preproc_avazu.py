import argparse
import os
import numpy as np
import torch

from recsys.datasets.avazu import CAT_FEATURE_COUNT, AvazuIterDataPipe, TOTAL_TRAINING_SAMPLES


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir",
                        type=str,
                        required=True,
                        help="path to the dir where the csv file train is downloaded and unzipped")

    parser.add_argument("--output_dir",
                        type=str,
                        required=True,
                        help="path to which the train/val/test splits are saved")

    parser.add_argument("--is_split", action='store_true')
    return parser.parse_args()


def main():
    # Note: this scripts is broken, to align with our experiments,
    # please refer to https://www.kaggle.com/code/leejunseok97/deepfm-deepctr-torch
    # Basically, the C14-C21 column of the resulting sparse files should be further split to the dense files.
    args = parse_args()

    if args.is_split:
        if not os.path.exists(args.input_dir):
            raise ValueError(f"{args.input_dir} has existed")

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        for _t in ("sparse", 'label'):
            npy = np.load(os.path.join(args.input_dir, f"{_t}.npy"))
            train_split = npy[:TOTAL_TRAINING_SAMPLES]
            np.save(os.path.join(args.output_dir, f"train_{_t}.npy"), train_split)
            val_test_split = npy[TOTAL_TRAINING_SAMPLES:]
            np.save(os.path.join(args.output_dir, f"val_test_{_t}.npy"), val_test_split)
            del npy

    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        sparse_output_file_path = os.path.join(args.output_dir, "sparse.npy")
        label_output_file_path = os.path.join(args.output_dir, "label.npy")

        sparse, labels = [], []
        for row_sparse, row_label in AvazuIterDataPipe(args.input_dir):
            sparse.append(row_sparse)
            labels.append(row_label)

        sparse_np = np.array(sparse, dtype=np.int32)
        del sparse
        labels_np = np.array(labels, dtype=np.int32).reshape(-1, 1)
        del labels

        for f_path, arr in [(sparse_output_file_path, sparse_np), (label_output_file_path, labels_np)]:
            np.save(f_path, arr)


if __name__ == "__main__":
    main()
