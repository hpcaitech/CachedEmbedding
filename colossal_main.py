from tqdm import tqdm
import itertools
from sklearn.metrics import roc_auc_score

import torch
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule

from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils import ColoInitContext
from colossalai.utils.cuda import get_current_device
from colossalai.tensor import ColoTensor, TensorSpec, ComputePattern, ParallelAction, DistSpecManager, distspec

from data.colossal_dataloader import get_dataloader
from utils import TrainValTestResults, trace_handler, get_mem_info
from models.colossal_dlrm import DLRM, reshape_spare_features


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
        prof=None
):
    logger = colossalai.logging.get_dist_logger()
    engine.train()
    data_iter = iter(data_loader)
    samples_per_trainer = len(data_loader) / gpc.get_world_size(ParallelMode.DATA) * epochs

    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            batch = next(data_iter).to(get_current_device())
            with record_function("Forward pass"):
                logits = engine(batch.dense_features, reshape_spare_features(batch.sparse_features)).squeeze()
            loss = engine.criterion(logits, batch.labels.float())

            if it == len(data_loader) - 3:
                logger.info(f"{get_mem_info('After forward:  ')}")

            engine.zero_grad()
            with record_function("Backward pass"):
                engine.backward(loss)

            with record_function("Optimization"):
                engine.step()

            prof.step()

            if change_lr and (
                (it * (epoch + 1) / samples_per_trainer) > lr_change_point
            ):  # progress made through the epoch
                logger.info(f"Changing learning rate to: {lr_after_change_point}", ranks=[0])
                change_lr = False
                optimizer = engine.optimizer
                lr = lr_after_change_point
                for g in optimizer.param_groups:
                    g["lr"] = lr
        except StopIteration:
            break


def _evaluate(
        engine,
        data_loader,
        stage
):
    engine.eval()
    # To enable torchmetrics,
    # modify colossalai/utils/model/colo_init_context.py line#72:
    #   change `ColoTensor` to `ColoParameter`
    auroc = metrics.AUROC(compute_on_step=False).to(get_current_device())
    accuracy = metrics.Accuracy(compute_on_step=False).to(get_current_device())
    data_iter = iter(data_loader)
    # pred_buffer, label_buffer = [], []
    with torch.no_grad():
        for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set:"):
            try:
                batch = next(data_iter).to(get_current_device())
                logits = engine(batch.dense_features, reshape_spare_features(batch.sparse_features)).squeeze().detach()
                preds = torch.sigmoid(logits)

                labels = batch.labels.detach()
                auroc(preds, labels)
                accuracy(preds, labels)
                # pred_buffer.extend(preds.tolist())
                # label_buffer.extend(labels.tolist())
            except StopIteration:
                break

    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    # valid_auc = roc_auc_score(label_buffer, pred_buffer)
    if gpc.get_global_rank() == 0:
        # print(f"Valid AUROC: {valid_auc}")
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Accuracy over {stage} set: {accuracy_result}.")
    return auroc_result, accuracy_result


def train_val_test(
        args,
        engine,
        train_dataloader,
        val_dataloader,
        test_dataloader,
):
    train_val_test_results = TrainValTestResults()
    # device = torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.DATA)}")
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0, warmup=len(train_dataloader) - 2, active=2, repeat=1
            ),
            profile_memory=True,
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
                prof
            )

            val_accuracy, val_auroc = _evaluate(
                engine,
                val_dataloader,
                "val"
            )

            train_val_test_results.val_accuracies.append(val_accuracy)
            train_val_test_results.val_aurocs.append(val_auroc)

        test_accuracy, test_auroc = _evaluate(
            engine,
            test_dataloader,
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
    logger.info(f"launch rank {gpc.get_global_rank()}, Done, "
                f"DP size: {gpc.get_world_size(ParallelMode.DATA)}, "
                f"TP size: {gpc.get_world_size(ParallelMode.PARALLEL_1D)}")

    train_dataloader = get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, "val")
    test_dataloader = get_dataloader(args, "test")

    logger.info(f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
                f"test batches: {len(test_dataloader)}", ranks=[0])

    with ColoInitContext(device=get_current_device()):
        model = DLRM(args.num_embeddings_per_feature, args.embedding_dim, len(DEFAULT_CAT_NAMES),
                     len(DEFAULT_INT_NAMES), list(map(int, args.dense_arch_layer_sizes.split(","))),
                     list(map(int, args.over_arch_layer_sizes.split(","))))
    logger.info(f"{get_mem_info('After model Init:  ')}")

    spec = TensorSpec(
        distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        ParallelAction(ComputePattern.TP1D))
    with DistSpecManager.no_grad():
        # Here only sets embedding to be model parallelized to align with torchrec
        model.sparse_arch.embed.weight.set_spec(spec)

    # TODO: check ColoOptimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    engine, _, _, _ = colossalai.initialize(model, optimizer, criterion)

    # data_iter = iter(train_dataloader)
    # batch = next(data_iter).to(get_current_device())
    # logits = engine(batch.dense_features, reshape_spare_features(batch.sparse_features)).squeeze()
    # logger.info(f"Logits: {logits}", ranks=[0])
    # loss = engine.criterion(logits, batch.labels.float())
    # logger.info(f"loss: {loss}", ranks=[0])
    # engine.backward(loss)
    # engine.step()
    # logger.info("Test Done", ranks=[0])
    # exit(0)

    train_val_test(
        args, engine, train_dataloader, val_dataloader, test_dataloader
    )


if __name__ == '__main__':
    main()
