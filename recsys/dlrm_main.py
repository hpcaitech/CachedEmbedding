from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
import itertools
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function
import torchmetrics as metrics

from recsys.utils import get_default_parser, get_mem_info
from recsys.datasets import criteo
from recsys import (disable_existing_loggers, launch_from_torch, ParallelMode, DISTMGR as dist_manager, DISTLogger as
                    dist_logger)
from recsys.models.dlrm import HybridParallelDLRM


def parse_args():
    parser = get_default_parser()

    # debug
    parser.add_argument('--profile_dir',
                        type=str,
                        default='tensorboard_log/recsys',
                        help='Specify the directory where profiler files are saved for tensorboard visualization')
    parser.add_argument('--inspect_time',
                        action='store_true',
                        help='Enable this option to inspect the overhead of a single iteration in the 5-th iteration, '
                        'instead of running the whole training process')
    parser.add_argument('--fused_op',
                        type=str,
                        default='all_to_all',
                        help='Specify the fused collective functions between Embedding and Dense, '
                        'permitted option: all_to_all | gather_scatter')

    # stress test
    parser.add_argument("--memory_fraction", type=float, default=None)
    parser.add_argument("--num_embeddings", type=int, default=10000)
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=None,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=None,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=None,
        help="number of test batches",
    )

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
    parser.add_argument("--use_cpu", action='store_true')
    parser.add_argument("--use_sparse_embed_grad", action='store_true')
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--cache_sets",
                        type=int,
                        default=500_000,
                        help="Number of cache sets in the cache. "
                        "*** Please make sure it can hold AT LEAST ONE BATCH OF SPARSE FEATURE IDS ***")
    parser.add_argument(
        "--cache_lines",
        type=int,
        default=1,
        help="Number of cache lines in each cache set. Similar to the N-way set associate mechanism in cache."
        "Not implemented yet. Increasing this would scale up the cache capacity")

    # Training
    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
        help="Random seed for reproducibility.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size to use for training")
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
        setattr(args, 'num_embeddings_per_feature', criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)
    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(map(int, args.num_embeddings_per_feature.split(",")))
    if args.in_memory_binary_criteo_path is None:
        for stage in criteo.STAGES:
            attr = f"limit_{stage}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    return args


def put_data_in_device(batch, dense_device, sparse_device):
    return batch.dense_features.to(dense_device), batch.sparse_features.to(sparse_device), batch.labels.to(dense_device)


@dataclass
class TrainValTestResults:
    val_accuracies: List[float] = field(default_factory=list)
    val_aurocs: List[float] = field(default_factory=list)
    test_accuracy: Optional[float] = None
    test_auroc: Optional[float] = None


def _train(model,
           optimizer,
           criterion,
           data_loader,
           epoch,
           epochs,
           change_lr,
           lr_change_point,
           lr_after_change_point,
           prof=None):
    model.train()
    data_iter = iter(data_loader)
    samples_per_trainer = criteo.KAGGLE_TOTAL_TRAINING_SAMPLES / dist_manager.get_world_size() * epochs
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            dense, sparse, labels = put_data_in_device(next(data_iter), model.dense_device, model.sparse_device)
            with record_function("(zhg)forward pass"):
                logits = model(dense, sparse).squeeze()

            loss = criterion(logits, labels.float())
            # dist_logger.info(f"loss: {loss.item()}")

            optimizer.zero_grad()
            with record_function("(zhg)backward pass"):
                loss.backward()

            with record_function("(zhg)optimization"):
                optimizer.step()

            if prof:
                prof.step()

            if change_lr and (it * (epoch + 1) / samples_per_trainer) > lr_change_point:
                dist_logger.info(f"Changing learning rate to: {lr_after_change_point}", ranks=[0])
                change_lr = False
                lr = lr_after_change_point
                for g in optimizer.param_groups:
                    g["lr"] = lr
        except StopIteration:
            dist_logger.info(f"{get_mem_info('Training:  ')}")
            break


def _evaluate(model, data_loader, stage):
    model.eval()
    auroc = metrics.AUROC(compute_on_step=False).cuda()
    accuracy = metrics.Accuracy(compute_on_step=False).cuda()

    data_iter = iter(data_loader)

    with torch.no_grad():
        for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set"):
            try:
                dense, sparse, labels = put_data_in_device(next(data_iter), model.dense_device, model.sparse_device)
                logits = model(dense, sparse).squeeze()
                preds = torch.sigmoid(logits)
                auroc(preds, labels)
                accuracy(preds, labels)
            except StopIteration:
                break

    auroc_res = auroc.compute().item()
    accuracy_res = accuracy.compute().item()
    dist_logger.info(f"AUROC over {stage} set: {auroc_res}", ranks=[0])
    dist_logger.info(f"Accuracy over {stage} set: {accuracy_res}", ranks=[0])
    return auroc_res, accuracy_res


def train_val_test(
    args,
    model,
    optimizer,
    criterion,
    train_dataloader,
    val_dataloader,
    test_dataloader,
):
    train_val_test_results = TrainValTestResults()
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=20, active=2, repeat=1),
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
    ) as prof:
        for epoch in range(args.epochs):
            _train(model, optimizer, criterion, train_dataloader, epoch, args.epochs, args.change_lr,
                   args.lr_change_point, args.lr_after_change_point, prof)

            val_accuracy, val_auroc = _evaluate(model, val_dataloader, "val")

            train_val_test_results.val_accuracies.append(val_accuracy)
            train_val_test_results.val_aurocs.append(val_auroc)

        test_accuracy, test_auroc = _evaluate(model, test_dataloader, "test")
        train_val_test_results.test_accuracy = test_accuracy
        train_val_test_results.test_auroc = test_auroc

    return train_val_test_results


def main():
    args = parse_args()
    disable_existing_loggers()

    launch_from_torch(backend='nccl', seed=args.seed)
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    dist_logger.info(f"launch rank: {dist_manager.get_rank()}, {dist_manager.get_distributed_info()}")
    dist_logger.info(f"config: {args}", ranks=[0])

    train_dataloader = criteo.get_dataloader(args, 'train')
    val_dataloader = criteo.get_dataloader(args, "val")
    test_dataloader = criteo.get_dataloader(args, "test")
    if args.in_memory_binary_criteo_path is not None:
        dist_logger.info(
            f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
            f"test batches: {len(test_dataloader)}",
            ranks=[0])

    device = torch.device('cuda', torch.cuda.current_device())
    sparse_device = torch.device('cpu') if args.use_cpu else device
    model = HybridParallelDLRM(
        [args.num_embeddings] *
        len(criteo.DEFAULT_CAT_NAMES) if args.in_memory_binary_criteo_path is None else args.num_embeddings_per_feature,
        args.embedding_dim,
        len(criteo.DEFAULT_CAT_NAMES),
        len(criteo.DEFAULT_INT_NAMES),
        list(map(int, args.dense_arch_layer_sizes.split(","))),
        list(map(int, args.over_arch_layer_sizes.split(","))),
        device,
        sparse_device,
        sparse=args.use_sparse_embed_grad,
        fused_op=args.fused_op,
        use_cache=args.use_cache,
        cache_sets=args.cache_sets,
        cache_lines=args.cache_lines,
    )
    dist_logger.info(f"{model.model_stats('DLRM')}", ranks=[0])
    dist_logger.info(f"{get_mem_info('After model init:  ')}", ranks=[0])
    for name, param in model.named_parameters():
        dist_logger.info(f"{name} : shape {param.shape}, device {param.data.device}", ranks=[0])

    rank = dist_manager.get_rank()
    world_size = dist_manager.get_world_size()
    # TODO: a more canonical interface for optimizers
    optimizer = torch.optim.SGD([{
        "params": model.sparse_modules.parameters(),
        "lr": args.learning_rate
    }, {
        "params": model.dense_modules.parameters(),
        "lr": args.learning_rate * world_size
    }])
    criterion = torch.nn.BCEWithLogitsLoss()

    if args.inspect_time:
        # Sanity check & iter time inspection
        from recsys.utils import get_time_elapsed

        data_iter = iter(train_dataloader)

        for i in range(5):
            batch = next(data_iter)
            optimizer.zero_grad()

            with get_time_elapsed(dist_logger, f"{i}-th data movement"):
                dense_features, sparse_features, labels = put_data_in_device(batch, device, sparse_device)
            # dist_logger.info(f"{i}-th sparse_features: {sparse_features.values()[:10]}")

            with get_time_elapsed(dist_logger, f"{i}-th forward pass"):
                logits = model(dense_features, sparse_features, inspect_time=True).squeeze()

            loss = criterion(logits, labels.float())
            dist_logger.info(f"{i}-th loss: {loss}")

            with get_time_elapsed(dist_logger, f"{i}-th backward pass"):
                loss.backward()

            with get_time_elapsed(dist_logger, f"{i}-th optimization"):
                optimizer.step()

        exit(0)

    train_val_test(args, model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
