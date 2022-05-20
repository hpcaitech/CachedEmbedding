from tqdm import tqdm
import itertools

import torch
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec import EmbeddingBagCollection
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.distributed import TrainPipelineSparseDist

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc

from data.colossal_dataloader import get_dataloader
from dlrm_main import TrainValTestResults, trace_handler
from models.dlrm import DLRM


def parse_args():
    parser = colossalai.get_default_parser()

    # ColossalAI config
    parser.add_argument('--config_path', default='./colossal_config.py', type=str)

    # Dataset
    parser.add_argument("--kaggle", action='store_true')
    parser.add_argument(
        "--pin_memory",
        dest="pin_memory",
        action="store_true",
        help="Use pinned memory when loading data.",
    )
    parser.add_argument(
        "--mmap_mode",
        dest="mmap_mode",
        action="store_true",
        help="--mmap_mode mmaps the dataset."
             " That is, the dataset is kept on disk but is accessed as if it were in memory."
             " --mmap_mode is intended mostly for faster debugging. Use --mmap_mode to bypass"
             " preloading the dataset when preloading takes too long or when there is "
             " insufficient memory available to load the full dataset.",
    )
    parser.add_argument(
        "--in_memory_binary_criteo_path",
        type=str,
        default=None,
        help="Path to a folder containing the binary (npy) files for the Criteo dataset."
             " When supplied, InMemoryBinaryCriteoIterDataPipe is used.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )

    # Model
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
             " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument(
        "--dense_arch_layer_sizes",
        type=str,
        default="512,256,64",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="512,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=64,
        help="Size of each embedding.",
    )

    # Training
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--change_lr",
        dest="change_lr",
        action="store_true",
        help="Flag to determine whether learning rate should be changed part way through training.",
    )
    parser.add_argument(
        "--lr_change_point",
        type=float,
        default=0.80,
        help="The point through training at which learning rate should change to the value set by"
             " lr_after_change_point. The default value is 0.80 which means that 80% through the total iterations (totaled"
             " across all epochs), the learning rate will change.",
    )
    parser.add_argument(
        "--lr_after_change_point",
        type=float,
        default=0.20,
        help="Learning rate after change point in first epoch.",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )

    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        shuffle_batches=None,
        change_lr=None,
    )

    args = parser.parse_args()

    if args.kaggle:
        from dlrm_main import NUM_EMBEDDINGS_PER_FEATURE
        global TOTAL_TRAINING_SAMPLES
        TOTAL_TRAINING_SAMPLES = 45840617
        setattr(args, 'num_embeddings_per_feature', NUM_EMBEDDINGS_PER_FEATURE)
    args.num_embeddings_per_feature = list(
        map(int, args.num_embeddings_per_feature.split(","))
    )

    return args


def _train(
        engine,
        data_loader,
        epoch,
        epochs,
        change_lr,
        lr_change_point,
        lr_after_change_point,
        device,
        prof=None
):
    engine.train()
    data_iter = iter(data_loader)
    samples_per_trainer = len(data_loader) / gpc.get_world_size(ParallelMode.DATA) * epochs

    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            if gpc.get_global_rank() == 0:
                print("pre")
            batch = next(data_iter).to(device)
            logits = engine(batch.dense_features, batch.sparse_features).squeeze()
            if gpc.get_global_rank() == 0:
                print("loss")
            loss = engine.criterion(logits, batch.labels.float())

            engine.zero_grad()
            engine.backward(loss)
            engine.step()

            prof.step()

            if change_lr and (
                (it * (epoch + 1) / samples_per_trainer) > lr_change_point
            ):  # progress made through the epoch
                print(f"Changing learning rate to: {lr_after_change_point}")
                optimizer = engine.optimizer
                lr = lr_after_change_point
                for g in optimizer.param_groups:
                    g["lr"] = lr
        except StopIteration:
            break


def _evaluate(
        engine,
        data_loader,
        device,
        stage
):
    engine.eval()
    auroc = metrics.AUROC(compute_on_step=False).to(device)
    accuracy = metrics.Accuracy(compute_on_step=False).to(device)
    data_iter = iter(data_loader)

    for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set:"):
        try:
            batch = next(data_iter).to(device)
            logits = engine(batch.dense_features, batch.sparse_features).squeeze().detach()
            preds = torch.sigmoid(logits)

            labels = batch.labels.detach()
            auroc(preds, labels)
            accuracy(preds, labels)
        except StopIteration:
            break

    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    if gpc.get_global_rank() == 0:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Accuracy over {stage} set: {accuracy_result}.")
    return auroc_result, accuracy_result


def train_val_test(
        args,
        engine,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device
):
    train_val_test_results = TrainValTestResults()
    # device = torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.DATA)}")
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0, warmup=len(train_dataloader) - 3, active=2, repeat=1
            ),
            on_trace_ready=trace_handler,
    ) as prof:
        for epoch in range(args.epochs):
            _train(
                engine,
                train_dataloader,
                epoch,
                args.epochs,
                args.change_lr,
                args.lr_change_point,
                args.lr_after_change_point,
                device,
                prof
            )

            val_accuracy, val_auroc = _evaluate(
                engine,
                val_dataloader,
                device,
                "val"
            )

            train_val_test_results.val_accuracies.append(val_accuracy)
            train_val_test_results.val_aurocs.append(val_auroc)

        test_accuracy, test_auroc = _evaluate(
            engine,
            test_dataloader,
            device,
            "test"
        )
        train_val_test_results.test_accuracy = test_accuracy
        train_val_test_results.test_auroc = test_auroc

    return train_val_test_results


def main():
    args = parse_args()

    colossalai.logging.disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config_path, verbose=False)

    logger = colossalai.logging.get_dist_logger()
    logger.info(f"launch rank {gpc.get_global_rank()}, Done, DP rank: {gpc.get_local_rank(ParallelMode.DATA)} / "
                f"{gpc.get_world_size(ParallelMode.DATA)}")

    train_dataloader = get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, "val")
    test_dataloader = get_dataloader(args, "test")

    logger.info(f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
                f"test batches: {len(test_dataloader)}", ranks=[0])

    # TODO: device rank in different groups
    device = torch.device(f"cuda:{gpc.get_global_rank()}")

    # TODO: check EmbeddingBagCollection
    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=args.num_embeddings_per_feature[feature_idx],
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

    # TODO: check Model
    train_model = DLRM(
        embedding_bag_collection=ebc,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,  # TODO: dense devices in the DP group
    )

    # TODO: check DistributedModelParallel
    fused_params = {
        "learning_rate": args.learning_rate,
        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD if args.adagrad else OptimType.EXACT_SGD,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]
    model = DistributedModelParallel(
        module=train_model,
        device=device,  # TODO: shard devices across the global group
        sharders=sharders,
    )
    logger.info(f"Model plan: {model.plan}", ranks=[0])

    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate)
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    # TODO: optimizer
    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),  # why dense? what about spares?
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    criterion = torch.nn.BCEWithLogitsLoss()

    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion)

    data_iter = iter(train_dataloader)
    batch = next(data_iter).to(device)
    logger.info(f"Batch: {batch}", ranks=[0])
    logits = engine(batch.dense_features, batch.sparse_features).squeeze()
    logger.info(f"Logits: {logits}", ranks=[0])
    loss = engine.criterion(logits, batch.labels.float())
    logger.info(f"loss: {loss}", ranks=[0])
    engine.backward(loss)
    engine.step()
    logger.info("Test Done", ranks=[0])
    exit(0)
    # TODO: pipeline
    # train_pipeline = TrainPipelineSparseDist(
    #     model,
    #     optimizer,
    #     device,
    # )
    train_val_test(
        args, engine, train_dataloader, val_dataloader, test_dataloader, device
    )


if __name__ == '__main__':
    main()
