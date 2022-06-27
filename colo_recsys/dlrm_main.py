import numpy as np
from tqdm import tqdm
import itertools
from sklearn.metrics import roc_auc_score

import torch
import torch.distributed as dist
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)

import colossalai
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils.cuda import get_current_device
from colossalai.tensor import ComputeSpec, ComputePattern, distspec, DistSpecManager, TensorSpec
from colossalai.nn.parallel import ColoDDP
from colossalai.nn.parallel.layers import init_colo_module

from colo_recsys.datasets.colossal_dataloader import get_dataloader
from colo_recsys.utils import TrainValTestResults, trace_handler, get_mem_info
from colo_recsys.models.colossal_dlrm import HybridParallelDLRM


def parse_args():
    parser = colossalai.get_default_parser()

    # tensorboard profiler
    parser.add_argument('--profile_dir', type=str, default='log/debug')

    # ColossalAI config
    parser.add_argument('--config_path', default='colo_recsys/dlrm_config.py', type=str)

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
        from baselines.dlrm_main import NUM_EMBEDDINGS_PER_FEATURE
        global TOTAL_TRAINING_SAMPLES
        TOTAL_TRAINING_SAMPLES = 45840617
        setattr(args, 'num_embeddings_per_feature', NUM_EMBEDDINGS_PER_FEATURE)
    args.num_embeddings_per_feature = list(map(int, args.num_embeddings_per_feature.split(",")))

    return args


def put_data_to_device(batch, use_cpu):
    device = get_current_device()
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)

    dense = torch.tensor_split(batch.dense_features.to(device), world_size, 0)[rank]
    labels = torch.tensor_split(batch.labels.to(device), world_size, 0)[rank]
    if use_cpu:
        sparse_features = batch.sparse_features
    else:
        sparse_features = batch.sparse_features.to(device)
    return dense, sparse_features, labels


def _train(
    model,
    criterion,
    optimizer,
    data_loader,
    epoch,
    epochs,
    change_lr,
    lr_change_point,
    lr_after_change_point,
    prof=None,
    use_cpu=False,
):
    logger = colossalai.logging.get_dist_logger()
    model.train()
    data_iter = iter(data_loader)
    samples_per_trainer = len(data_loader) / gpc.get_world_size(ParallelMode.DATA) * epochs

    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            dense, sparse, labels = put_data_to_device(next(data_iter), use_cpu)
            with record_function("Forward pass"):
                logits = model(dense, sparse).squeeze()
            loss = criterion(logits, labels.float())
            logger.info(f"it: {it}, loss: {loss.item()}")
            if it == len(data_loader) - 3:
                logger.info(f"{get_mem_info('After forward:  ')}")

            optimizer.zero_grad()
            with record_function("Backward pass"):
                model.backward(loss)

            with record_function("Optimization"):
                optimizer.step()

            if prof:
                prof.step()

            if change_lr and (
                (it * (epoch + 1) / samples_per_trainer) > lr_change_point):    # progress made through the epoch
                logger.info(f"Changing learning rate to: {lr_after_change_point}", ranks=[0])
                change_lr = False
                lr = lr_after_change_point
                for g in optimizer.param_groups:
                    g["lr"] = lr
        except StopIteration:
            break


def _evaluate(model, data_loader, stage, use_cpu=False):
    model.eval()
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
                dense, sparse, labels = put_data_to_device(next(data_iter), use_cpu)
                logits = model(dense, sparse).squeeze().detach()
                preds = torch.sigmoid(logits)

                labels = labels.detach()
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
    model,
    criterion,
    optimizer,
    train_dataloader,
    val_dataloader,
    test_dataloader,
):
    train_val_test_results = TrainValTestResults()
    # device = torch.device(f"cuda:{gpc.get_local_rank(ParallelMode.DATA)}")
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=20, active=2, repeat=1),
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
    ) as prof:
        for epoch in range(args.epochs):
            _train(model, criterion, optimizer, train_dataloader, epoch, args.epochs, args.change_lr,
                   args.lr_change_point, args.lr_after_change_point, prof, args.use_cpu)

            val_accuracy, val_auroc = _evaluate(model, val_dataloader, "val", args.use_cpu)

            train_val_test_results.val_accuracies.append(val_accuracy)
            train_val_test_results.val_aurocs.append(val_auroc)

        test_accuracy, test_auroc = _evaluate(model, test_dataloader, "test", args.use_cpu)
        train_val_test_results.test_accuracy = test_accuracy
        train_val_test_results.test_auroc = test_auroc

    return train_val_test_results


def main():
    args = parse_args()

    colossalai.logging.disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config_path, verbose=False, seed=args.seed)

    logger = colossalai.logging.get_dist_logger()
    logger.info(f"[DEBUG] launch rank {gpc.get_global_rank()}, "
                f"DP size: {gpc.data_parallel_size}, "
                f"TP size: {gpc.tensor_parallel_size}")

    train_dataloader = get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, "val")
    test_dataloader = get_dataloader(args, "test")

    logger.info(
        f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
        f"test batches: {len(test_dataloader)}",
        ranks=[0])

    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    group = gpc.get_group(ParallelMode.PARALLEL_1D)
    device = get_current_device()
    embed_ouptut_device = 'cuda' if args.use_cpu else None
    with ColoInitContext(device=device):
        model = HybridParallelDLRM(args.num_embeddings_per_feature,
                                   args.embedding_dim,
                                   len(DEFAULT_CAT_NAMES),
                                   len(DEFAULT_INT_NAMES),
                                   list(map(int, args.dense_arch_layer_sizes.split(","))),
                                   list(map(int, args.over_arch_layer_sizes.split(","))),
                                   output_device_type=embed_ouptut_device)
    logger.info(f"{get_mem_info('After model Init:  ')}", ranks=[0])
    compute_spec = ComputeSpec(ComputePattern.TP1D)
    compute_spec.output_replicate = False
    # init_colo_module(model.sparse_modules, compute_spec, recursive=True, mode='col')
    spec = TensorSpec(distspec.shard(group, [-1], [world_size]), compute_spec)
    with DistSpecManager.no_grad():
        model.sparse_modules.embed.weight.set_tensor_spec(spec)
    if args.use_cpu:
        model.sparse_modules.to('cpu')
        if gpc.tensor_parallel_size > 1:
            gloo_group_tp = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
            model.sparse_modules.embed.weight.tensor_spec.dist_spec.process_group = gloo_group_tp

    ignore_param = [v for k, v in model.named_parameters() if 'embed' in k]
    ColoDDP.set_params_to_ignore(ignore_param)
    model = ColoDDP(model, group)

    logger.info(f"{get_mem_info('After colossalai init:  ')}", ranks=[0])
    for name, param in model.named_parameters():
        logger.info(f"{name} : shape {param.shape}, spec: {param.tensor_spec.dist_spec}", ranks=[0])

    optim_cls = gpc.config.optimizer.pop('type')
    optimizer = optim_cls([{
        "params": model.module.sparse_modules.parameters(),
        "lr": args.learning_rate
    }, {
        "params": model.module.dense_modules.parameters(),
        "lr": args.learning_rate * gpc.get_world_size(ParallelMode.PARALLEL_1D)
    }])
    criterion = torch.nn.BCEWithLogitsLoss()

    # Sanity Check & Time inspection
    if gpc.config.inspect_time:
        from colo_recsys.utils import get_time_elapsed

        data_iter = iter(train_dataloader)

        for i in range(10):
            batch = next(data_iter)
            with get_time_elapsed(logger, f"{i}-th data movement"):
                dense_features, sparse_features, labels = put_data_to_device(batch, args.use_cpu)

            # logger.info(f"rank: {gpc.get_local_rank(ParallelMode.PARALLEL_1D)}, {i}-th iter "
            #             f"sparse: {sparse_features.values()[:10].tolist()}")

            with get_time_elapsed(logger, f"{i}-th forward pass"):
                logits = model(dense_features, sparse_features).squeeze()

            loss = criterion(logits, labels.float())
            logger.info(f"{i}-th loss: {loss}", ranks=[0])
            optimizer.zero_grad()
            with get_time_elapsed(logger, f"{i}-th backward pass"):
                model.backward(loss)

            with get_time_elapsed(logger, f"{i}-th optimization"):
                optimizer.step()
        gpc.destroy()
        exit(0)

    train_val_test(args, model, criterion, optimizer, train_dataloader, val_dataloader, test_dataloader)


if __name__ == '__main__':
    main()
