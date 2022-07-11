import itertools
import os
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)

from persia.utils import setup_seed
from persia.logger import get_default_logger
from persia.env import get_rank, get_local_rank, get_world_size
from persia.embedding.optim import SGD
from persia.data import DataLoader, IterableDataset, StreamingDataset
from persia.ctx import TrainCtx, eval_ctx
from persia.embedding.data import PersiaBatch

from models.dlrm import Dense
from data.dlrm_dataloader import get_persia_dataloader, KAGGLE_NUM_EMBEDDINGS_PER_FEATURE

logger = get_default_logger("nn_worker")
EPOCHS = 1
BUFFER_SIZE = 10


class TestDataset(IterableDataset):

    def __init__(self,
                 test_dir,
                 batch_size,
                 stage,
                 buffer_size=BUFFER_SIZE,
                 num_embeddings_per_feature=None,
                 seed=None,
                 num_batches=None):
        super(TestDataset, self).__init__(buffer_size=buffer_size)
        self.loader = get_persia_dataloader(stage,
                                            test_dir,
                                            batch_size,
                                            num_embeddings_per_feature=num_embeddings_per_feature,
                                            seed=seed,
                                            num_batches=num_batches)

    def __iter__(self):
        for dense_features, sparse_features, labels in self.loader:
            yield PersiaBatch(
                sparse_features,
                non_id_type_features=[dense_features],
                labels=[labels],
                requires_grad=False,
            )


def _train(train_ctx, criterion, train_iter, epoch):
    for it in tqdm(itertools.count(), desc=f"Epoch: {epoch}"):
        try:
            logger.info(f"iter: {it}")
            batch = next(train_iter)
            (output, labels) = train_ctx.forward(batch)
            label = labels[0].squeeze(1).float()
            loss = criterion(output.squeeze(), label)
            train_ctx.backward(loss)
        except StopIteration:
            break


def _evaluate(model, data_iter, stage):
    logger.info(f"Start to {stage}")
    model.eval()
    pred_buffer, label_buffer = [], []
    with eval_ctx(model=model) as ctx:
        for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set:"):
            try:
                batch = next(data_iter)
                (output, labels) = ctx.forward(batch)
                label = labels[0].squeeze(1)
                preds = torch.sigmoid(output.squeeze())
                pred_buffer.extend(preds.tolist())
                label_buffer.extend(label.tolist())
            except StopIteration:
                break
    auroc = roc_auc_score(label_buffer, pred_buffer)
    logger.info(f"AUROC over {stage} set: {auroc}")

    model.train()


def main():
    # Env setup
    seed = int(os.environ.get("REPRODUCIBLE", 0))
    reproducible = bool(seed)
    embedding_staleness = int(os.environ.get("EMBEDDING_STALENESS", 10))
    dataset = os.environ.get("DATASET_DIR", None)
    batch_size = int(os.environ.get("BATCH_SIZE", 16384))
    num_embeddings_per_feature = os.environ.get("num_embeddings_per_feature".upper(), KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)

    num_val_batches = int(os.environ.get("limit_val_batches".upper(), 10))
    num_test_batches = int(os.environ.get("limit_test_batches".upper(), 10))
    if reproducible:
        setup_seed(seed)

    rank, device_id, world_size = get_rank(), get_local_rank(), get_world_size()
    torch.cuda.set_device(device_id)
    logger.info(f"device id is {device_id}")

    # Model
    # confused about how to specify customized arguments outside the persia-launcher
    model = Dense(embedding_dim=128,
                  num_sparse_features=len(DEFAULT_CAT_NAMES),
                  dense_in_features=len(DEFAULT_INT_NAMES),
                  dense_arch_layer_sizes=[512, 256, 128],
                  over_arch_layer_sizes=[1024, 1024, 512, 256, 1]).cuda()
    logger.info("DLRM is initialized")

    # Optimizer
    dense_optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    embedding_optimizer = SGD(lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    # Val & Test data
    logger.info(f"debug: {num_embeddings_per_feature}")
    val_dataset = TestDataset(dataset,
                              batch_size=batch_size,
                              stage='val',
                              num_embeddings_per_feature=num_embeddings_per_feature,
                              seed=seed if reproducible else None,
                              num_batches=num_val_batches)
    test_dataset = TestDataset(dataset,
                               batch_size=batch_size,
                               stage='test',
                               num_embeddings_per_feature=num_embeddings_per_feature,
                               seed=seed if reproducible else None,
                               num_batches=num_test_batches)

    with TrainCtx(model=model,
                  embedding_optimizer=embedding_optimizer,
                  dense_optimizer=dense_optimizer,
                  mixed_precision=True,
                  device_id=device_id) as ctx:
        train_dataloader = DataLoader(
            StreamingDataset(BUFFER_SIZE),
            reproducible=reproducible,
            embedding_staleness=embedding_staleness,
        )
        val_dataloader = DataLoader(val_dataset)
        test_dataloader = DataLoader(test_dataset)

        logger.info(f"Start training")
        for epoch in range(EPOCHS):
            _train(ctx, loss_fn, iter(train_dataloader), epoch)

            _evaluate(model, iter(val_dataloader), "val")
        _evaluate(model, iter(test_dataloader), "test")


if __name__ == "__main__":
    main()
