import os

from torchrec.datasets.criteo import (
    CAT_FEATURE_COUNT,
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    DAYS,
    InMemoryBinaryCriteoIterDataPipe,
)
from torchrec.datasets.random import RandomRecDataset
from torch.utils.data import DataLoader

from .. import ParallelMode, DISTMGR

STAGES = ["train", "val", "test"]

KAGGLE_NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,' \
                                           '27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'
KAGGLE_TOTAL_TRAINING_SAMPLES = 39291954    # 0-6 days for criteo kaggle, 45840617 samples in total


def get_dataloader(args, stage, parallel_mode: ParallelMode = ParallelMode.DEFAULT):
    stage = stage.lower()
    if stage not in STAGES:
        raise ValueError(f"Supplied stage was {stage}. Must be one of {STAGES}.")

    args.pin_memory = (args.backend == "nccl") if not hasattr(args, "pin_memory") else args.pin_memory

    if args.in_memory_binary_criteo_path is None:
        return DataLoader(
            RandomRecDataset(
                keys=DEFAULT_CAT_NAMES,
                batch_size=args.batch_size,
                hash_size=args.num_embeddings,
                hash_sizes=args.num_embeddings_per_feature if hasattr(args, "num_embeddings_per_feature") else None,
                manual_seed=args.seed if hasattr(args, "seed") else None,
                ids_per_feature=1,
                num_dense=len(DEFAULT_INT_NAMES),
                num_batches=getattr(args, f"limit_{stage}_batches")),
            batch_size=None,
            batch_sampler=None,
            pin_memory=args.pin_memory,
            num_workers=0,
        )

    files = os.listdir(args.in_memory_binary_criteo_path)

    def is_final_day(s: str) -> bool:
        return f"day_{(7 if args.kaggle else DAYS) - 1}" in s

    rank = DISTMGR.get_rank(parallel_mode)
    world_size = DISTMGR.get_world_size(parallel_mode)
    if stage == "train":
        # Train set gets all data except from the final day.
        files = list(filter(lambda s: not is_final_day(s), files))
    else:
        # Validation set gets the first half of the final day's samples. Test set get
        # the other half.
        files = list(filter(is_final_day, files))
        rank = rank if stage == "val" else (rank + world_size)
        world_size = world_size * 2

    stage_files = [
        sorted(map(
            lambda x: os.path.join(args.in_memory_binary_criteo_path, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]

    dataloader = DataLoader(
        InMemoryBinaryCriteoIterDataPipe(
            *stage_files,    # pyre-ignore[6]
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=args.shuffle_batches,
            hashes=args.num_embeddings_per_feature),
        batch_size=None,
        pin_memory=args.pin_memory,
        collate_fn=lambda x: x,
    )
    return dataloader
