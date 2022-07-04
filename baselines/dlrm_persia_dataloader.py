import os

from tqdm import tqdm

from persia.ctx import DataCtx
from persia.embedding.data import PersiaBatch
from persia.logger import get_default_logger

from data.dlrm_dataloader import get_persia_dataloader, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE

logger = get_default_logger("dataloader")

if __name__ == "__main__":
    dataset = os.environ.get("DATASET_DIR", None)
    batch_size = int(os.environ.get("BATCH_SIZE", 16384))

    # if dataset is None, options below would take effect to generate random batches
    num_embeddings_per_feature = os.environ.get("num_embeddings_per_feature".upper(), KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)

    seed = int(os.environ.get("REPRODUCIBLE", 0))
    if seed == 0:
        seed = None

    num_batches = int(os.environ.get("limit_train_batches".upper(), 2000))

    with DataCtx() as ctx:
        loader = get_persia_dataloader('train',
                                       dataset,
                                       batch_size,
                                       num_embeddings_per_feature=num_embeddings_per_feature,
                                       seed=seed,
                                       num_batches=num_batches)
        logger.info(f"There are {len(loader) if dataset is not None else num_batches} batches")
        for dense_features, sparse_features, labels in tqdm(loader, desc="Generate batch..."):
            persia_batch = PersiaBatch(sparse_features,
                                       non_id_type_features=[dense_features],
                                       labels=[labels],
                                       requires_grad=True)
            ctx.send_data(persia_batch)
