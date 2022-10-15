import os
import time
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
import itertools
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler, record_function
import torchmetrics as metrics

from recsys.utils import get_mem_info
from recsys.datasets import criteo, avazu
from recsys.models.dlrm import HybridParallelDLRM
from recsys.utils import FiniteDataIter, TrainValTestResults

import colossalai

from utils import get_tablewise_rank_arrange

dist_logger = colossalai.logging.get_dist_logger()


def parse_args():
    parser = colossalai.get_default_parser()

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
        "--limit_train_samples",
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
        "--dataset_dir",
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
        default="512,256,128",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument(
        "--over_arch_layer_sizes",
        type=str,
        default="1024,1024,512,256,1",
        help="Comma separated layer sizes for over arch.",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=128,
        help="Size of each embedding.",
    )
    parser.add_argument("--use_cpu", action='store_true')
    parser.add_argument("--use_cache_mgr_async_copy", action='store_true')
    parser.add_argument("--use_sparse_embed_grad", action='store_true')
    parser.add_argument("--use_cache", action='store_true')
    parser.add_argument("--cache_ratio",
                        type=float,
                        default=0.01,
                        help="cache ratio. "
                        "*** Please make sure it can hold AT LEAST ONE BATCH OF SPARSE FEATURE IDS ***")
    parser.add_argument("--use_freq", action='store_true',
                        help="use the dataset freq information to initialize the softwar cache")
    parser.add_argument("--use_lfu", action='store_true',
                        help="use the LFU as the cache eviction strategy. If false use DATASET aware version")
    parser.add_argument("--warmup_ratio", type=float, default=0.7, help="warmup ratio of the software cache")
    parser.add_argument("--buffer_size", type=int, default=0,
                        help="limit buffer size, if buffer_size=1, do not use the buffer.")
    parser.add_argument("--use_tablewise", action='store_true',
                        help="use tablewise structure for cached embedding bag. If false use columnwise.")
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
        "--prefetch_num",
        type=int,
        default=1,
        help="Number of batch prefetched for caching",
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    parser.add_argument("--use_overlap", action="store_true")
    parser.add_argument("--use_distributed_dataloader", action="store_true")
    parser.add_argument(
        "--eval_acc",
        action="store_true",
        help="Evaluation the model accuracy on test dataset.",
    )

    args = parser.parse_args()

    if args.dataset_dir is not None:
        if 'criteo' in args.dataset_dir:
            if 'kaggle' in args.dataset_dir:
                setattr(args, 'num_embeddings_per_feature', criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)
            else:
                setattr(args, 'num_embeddings_per_feature', criteo.NUM_EMBEDDINGS_PER_FEATURE)
        elif 'avazu' in args.dataset_dir:
            setattr(args, 'num_embeddings_per_feature', avazu.NUM_EMBEDDINGS_PER_FEATURE)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(map(int, args.num_embeddings_per_feature.split(",")))
    if args.dataset_dir is None:
        for stage in criteo.STAGES:
            attr = f"limit_{stage}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    return args


def put_data_in_device(batch, dense_device, sparse_device, is_dist=False, rank=0, world_size=1):
    if is_dist:
        return batch.dense_features.to(dense_device), batch.sparse_features.to(sparse_device), batch.labels.to(
            dense_device)
    else:
        dense_features = torch.tensor_split(batch.dense_features.to(dense_device), world_size, dim=0)[rank]
        labels = torch.tensor_split(batch.labels.to(dense_device), world_size, dim=0)[rank]
        sparse_features = batch.sparse_features.to(sparse_device)
        return dense_features, sparse_features, labels

    
def _train(model : HybridParallelDLRM,
           optimizer,
           criterion,
           data_loader,
           epoch,
           prof=None,
           use_overlap=True,
           use_distributed_dataloader=True,
           prefetch_num = 1):
    model.train()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if use_overlap:
        data_iter = FiniteDataIter(data_loader)
    else:
        data_iter = iter(data_loader)

    total = len(data_loader) if hasattr(data_loader, "__len__") else None
    meter = tqdm(itertools.count(), desc=f"Epoch {epoch}", ncols=0, total=total)
    time_elapse = 0.


    dense_list = [None for i in range(prefetch_num)]
    sparse_list = [None for i in range(prefetch_num)]
    labels_list = [None for i in range(prefetch_num)]
    sparse_values = [None for i in range(prefetch_num)]

    print(f'prefetch_num {prefetch_num}')
    for idx in meter:
        try:
            # We introduce a timer as a temporary solution to exclude interference
            # due to the bugs exists in NVTabular dataloader, please see my discussion:
            # https://github.com/dask/dask/discussions/9405.
            
            start = time.time()

            # trigger cache operations very prefetch num iterations.
            # prefetch #prefetch_num batches.
            prefetch_idx = idx % prefetch_num
            if prefetch_idx == 0:
                with torch.no_grad():
                    for i in range(prefetch_num):
                        batch = next(data_iter)
                        dense, sparse, labels = put_data_in_device(batch, model.dense_device, model.sparse_device,
                                                                use_distributed_dataloader, rank, world_size)
                        dense_list[i] = dense
                        sparse_list[i] = [sparse.values(), sparse.offsets(), sparse.stride()]
                        sparse_values[i] = sparse.values()
                        labels_list[i] = labels

                    
                    with record_function("prefetch cache"):
                        cuda_sparse_ids = model.sparse_modules.embed.cache_weight_mgr.prepare_ids(torch.cat(sparse_values))
                        cuda_sparse_list = torch.chunk(cuda_sparse_ids, prefetch_num)
                        for i in range(prefetch_num):
                            sparse_list[i][0] = cuda_sparse_list[i]
            
            dense = dense_list[prefetch_idx]
            sparse = sparse_list[prefetch_idx]
            labels = labels_list[prefetch_idx]

            with record_function("(zhg)forward pass"):
                logits = model(dense, sparse, cache_op = False).squeeze()

            loss = criterion(logits, labels.float())
            # dist_logger.info(f"loss: {loss.item()}")

            optimizer.zero_grad()
            with record_function("(zhg)backward pass"):
                loss.backward()

            with record_function("(zhg)optimization"):
                optimizer.step()
            time_elapse += time.time() - start
            if prof:
                prof.step()

            # Below will introduce additional overhead
            # postfix_str = f"loss={loss.item():.4f}"
            # if hasattr(model.sparse_modules.embed, "num_miss_history"):
            #     hit_rate = model.sparse_modules.embed.num_hits_history[-1] / (
            #         model.sparse_modules.embed.num_hits_history[-1] +
            #         model.sparse_modules.embed.num_miss_history[-1])
            #     postfix_str += f" hit rate={hit_rate*100:.2f}%"
            # meter.set_postfix_str(postfix_str)
        except StopIteration:
            dist_logger.info(f"{get_mem_info('Training:  ')}, "
                             f"{model.sparse_modules.embed.print_comm_stats_()}")
            break
    if hasattr(data_loader, "__len__"):
        dist_logger.info(f"average throughput: {len(data_loader) / time_elapse:.2f} it/s")


def _evaluate(model, data_loader, stage, use_overlap, use_distributed_dataloader):
    model.eval()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    auroc = metrics.AUROC(compute_on_step=False).cuda()
    accuracy = metrics.Accuracy(compute_on_step=False).cuda()

    if use_overlap:
        data_iter = FiniteDataIter(data_loader)
    else:
        data_iter = iter(data_loader)

    with torch.no_grad():
        for _ in tqdm(itertools.count(),
                      desc=f"Evaluating {stage} set",
                      ncols=0,
                      total=len(data_loader) if hasattr(data_loader, "__len__") else None):
            try:
                dense, sparse, labels = put_data_in_device(next(data_iter), model.dense_device, model.sparse_device,
                                                           use_distributed_dataloader, rank, world_size)
                logits = model(dense, sparse).squeeze()
                preds = torch.sigmoid(logits)
                # dist_logger.info(f"pred: {preds.max(), preds.min()}")
                labels = labels.int()
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
            schedule=schedule(wait=0, warmup=200, active=5, repeat=1),
            profile_memory=True,
            on_trace_ready=tensorboard_trace_handler(args.profile_dir),
    ) as prof:
        for epoch in range(args.epochs):

            model.sparse_modules.embed.set_cache_mgr_async_copy(args.use_cache_mgr_async_copy)
            _train(model, optimizer, criterion, train_dataloader, epoch, prof, args.use_overlap,
                   args.use_distributed_dataloader, prefetch_num=args.prefetch_num)

            if args.eval_acc:
                val_accuracy, val_auroc = _evaluate(model, val_dataloader, "val", args.use_overlap,
                                                    args.use_distributed_dataloader)

                train_val_test_results.val_accuracies.append(val_accuracy)
                train_val_test_results.val_aurocs.append(val_auroc)

        if args.eval_acc:
            test_accuracy, test_auroc = _evaluate(model, test_dataloader, "test", args.use_overlap,
                                                args.use_distributed_dataloader)
            train_val_test_results.test_accuracy = test_accuracy
            train_val_test_results.test_auroc = test_auroc

    return train_val_test_results


def main():
    args = parse_args()

    colossalai.logging.disable_existing_loggers()
    colossalai.launch_from_torch(config={}, seed=args.seed, verbose=False)

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)

    dataloader_factory = {"rank": 0, "world_size": 1}
    if args.use_distributed_dataloader:
        dataloader_factory["rank"] = rank
        dataloader_factory["world_size"] = world_size

    dist_logger.info(f"launch rank: {rank} / {world_size}")
    dist_logger.info(f"config: {args}", ranks=[0])

    assigned_tables = None
    if args.use_tablewise:
        rank_arrange = None
        rank_arrange = get_tablewise_rank_arrange(args.dataset_dir, torch.distributed.get_world_size())
        assigned_tables = []
        for table, assign_rank in enumerate(rank_arrange):
            if assign_rank == torch.distributed.get_rank():
                assigned_tables.append(table)
    
    if 'criteo' in args.dataset_dir:
        data_module = criteo
    elif 'avazu' in args.dataset_dir:
        data_module = avazu
    else:
        raise NotImplementedError()    # TODO: random data interface

    train_dataloader = data_module.get_dataloader(args, 'train', **dataloader_factory, assigned_tables = assigned_tables)
    val_dataloader = data_module.get_dataloader(args, "val", **dataloader_factory, assigned_tables = assigned_tables)
    test_dataloader = data_module.get_dataloader(args, "test", **dataloader_factory, assigned_tables = assigned_tables)

    if args.dataset_dir is not None and hasattr(train_dataloader, "__len__"):
        dist_logger.info(
            f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
            f"test batches: {len(test_dataloader)}",
            ranks=[0])

    id_freq_map = None
    if args.use_freq:
        id_freq_map = data_module.get_id_freq_map(args.dataset_dir)

    device = torch.device('cuda', torch.cuda.current_device())
    sparse_device = torch.device('cpu') if args.use_cpu else device
    model = HybridParallelDLRM(
        [args.num_embeddings]
        * len(data_module.DEFAULT_CAT_NAMES) if args.dataset_dir is None else args.num_embeddings_per_feature,
        args.embedding_dim,
        len(data_module.DEFAULT_CAT_NAMES),
        len(data_module.DEFAULT_INT_NAMES),
        list(map(int, args.dense_arch_layer_sizes.split(","))),
        list(map(int, args.over_arch_layer_sizes.split(","))),
        device,
        sparse_device,
        sparse=args.use_sparse_embed_grad,
        fused_op=args.fused_op,
        use_cache=args.use_cache,
        cache_ratio=args.cache_ratio,
        id_freq_map=id_freq_map,
        warmup_ratio=args.warmup_ratio,
        buffer_size=args.buffer_size,
        is_dist_dataloader=args.use_distributed_dataloader,
        use_lfu_eviction=args.use_lfu,
        use_tablewise=args.use_tablewise,
        dataset=args.dataset_dir
    )
    dist_logger.info(f"{model.model_stats('DLRM')}", ranks=[0])
    dist_logger.info(f"{get_mem_info('After model init:  ')}", ranks=[0])
    for name, param in model.named_parameters():
        dist_logger.info(f"{name} : shape {param.shape}, device {param.data.device}", ranks=[0])

    # TODO: a more canonical interface for optimizers.
    # currently not support ADAM
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

        data_iter = iter(train_dataloader)

        for i in range(200):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_dataloader)
                batch = next(data_iter)

            optimizer.zero_grad()

            # with get_time_elapsed(dist_logger, f"{i}-th data movement"):
            dense_features, sparse_features, labels = put_data_in_device(batch, device, sparse_device,
                                                                         args.use_distributed_dataloader, rank,
                                                                         world_size)
            # dist_logger.info(f"{i}-th sparse_features: {sparse_features.values()[:10]}")

            # with get_time_elapsed(dist_logger, f"{i}-th forward pass"):
            logits = model(dense_features, sparse_features, inspect_time=False).squeeze()

            loss = criterion(logits, labels.float())
            dist_logger.info(f"{i}-th loss: {loss}, logits: {logits}, labels: {labels}")

            # with get_time_elapsed(dist_logger, f"{i}-th backward pass"):
            loss.backward()

            # with get_time_elapsed(dist_logger, f"{i}-th optimization"):
            optimizer.step()
        exit(0)

    train_val_test(args, model, optimizer, criterion, train_dataloader, val_dataloader, test_dataloader)


if __name__ == "__main__":
    main()
