import os
from torch.utils.data import DataLoader
from recsys.datasets import criteo

CRITEO_PATH = "../criteo_kaggle_data"
NUM_EMBED = 33762577


def get_dataloader(stage, batch_size):
    hash_sizes = list(map(int, criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE.split(',')))
    files = os.listdir(CRITEO_PATH)

    def is_final_day(s):
        return "day_6" in s

    rank, world_size = 0, 1
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
            lambda x: os.path.join(CRITEO_PATH, x),
            filter(lambda s: kind in s, files),
        )) for kind in ["dense", "sparse", "labels"]
    ]
    dataloader = DataLoader(
        criteo.InMemoryBinaryCriteoIterDataPipe(
            *stage_files,    # pyre-ignore[6]
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            shuffle_batches=True,
            hashes=hash_sizes),
        batch_size=None,
        pin_memory=True,
        collate_fn=lambda x: x,
    )
    return dataloader
