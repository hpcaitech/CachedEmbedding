#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import os
import sys
from typing import cast, Iterator, List, Optional, Tuple

import torch
import torchmetrics as metrics
from pyre_extensions import none_throws
from torch import nn, distributed as dist
from torch.utils.data import DataLoader
from torchrec import EmbeddingBagCollection
from torchrec.datasets.criteo import (
    DEFAULT_CAT_NAMES,
    DEFAULT_INT_NAMES,
    TOTAL_TRAINING_SAMPLES,
)
from torchrec.datasets.utils import Batch
from torchrec.distributed import TrainPipelineSparseDist
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.model_parallel import DistributedModelParallel
from torchrec.distributed.types import ModuleSharder
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from tqdm import tqdm
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType

from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.distributed.types import ShardingEnv, ShardingType
from torchrec.distributed.planner.types import ParameterConstraints

from torch.profiler import profile, record_function, ProfilerActivity, schedule

# OSS import
try:
    # pyre-ignore[21]
    # @manual=//pytorch/benchmark/torchrec_dlrm/data:dlrm_dataloader
    from data.dlrm_dataloader import get_dataloader, STAGES

    # pyre-ignore[21]
    # @manual=//pytorch/benchmark/torchrec_dlrm/modules:dlrm_train
    from modules.dlrm_train import DLRMTrain
except ImportError:
    pass

# internal import
try:
    from .data.dlrm_dataloader import (  # noqa F811
        get_dataloader,
        STAGES,
    )
    from .modules.dlrm_train import DLRMTrain  # noqa F811
except ImportError:
    pass

from utils import TrainValTestResults, get_mem_info, trace_handler

TRAIN_PIPELINE_STAGES = 3  # Number of stages in TrainPipelineSparseDist.
NUM_EMBEDDINGS_PER_FEATURE = '1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,' \
                             '27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572'  # For criteo kaggle


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torchrec dlrm example trainer")
    parser.add_argument("--kaggle", action='store_true')
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
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
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="criteo_1t",
        help="dataset for experiment, current support criteo_1tb, criteo_kaggle",
    )
    parser.add_argument(
        "--num_embeddings",
        type=int,
        default=100_000,
        help="max_ind_size. The number of embeddings in each embedding table. Defaults"
        " to 100_000 if num_embeddings_per_feature is not supplied.",
    )
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
    parser.add_argument(
        "--undersampling_rate",
        type=float,
        help="Desired proportion of zero-labeled samples to retain (i.e. undersampling zero-labeled rows)."
        " Ex. 0.3 indicates only 30pct of the rows with label 0 will be kept."
        " All rows with label 1 will be kept. Value should be between 0 and 1."
        " When not supplied, no undersampling occurs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility.",
    )
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
        "--learning_rate",
        type=float,
        default=15.0,
        help="Learning rate.",
    )
    parser.add_argument(
        "--shuffle_batches",
        dest="shuffle_batches",
        action="store_true",
        help="Shuffle each batch during training.",
    )
    parser.add_argument(
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
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
    parser.set_defaults(
        pin_memory=None,
        mmap_mode=None,
        shuffle_batches=None,
        change_lr=None,
    )
    parser.add_argument(
        "--adagrad",
        dest="adagrad",
        action="store_true",
        help="Flag to determine if adagrad optimizer should be used.",
    )
    return parser.parse_args(argv)


def _evaluate(
    limit_batches: Optional[int],
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    stage: str,
    prof=None
) -> Tuple[float, float]:
    """
    Evaluates model. Computes and prints metrics including AUROC and Accuracy. Helper
    function for train_val_test.

    Args:
        limit_batches (Optional[int]): number of batches.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for val/test batches.
        next_iterator (Iterator[Batch]): Iterator used for the next phase (either train
            if there are more epochs to train on or test if all epochs are complete).
            Used to queue up the next TRAIN_PIPELINE_STAGES - 1 batches before
            train_val_test switches to the next phase. This is done so that when the
            next phase starts, the first output train_pipeline generates an output for
            is the 1st batch for that phase.
        stage (str): "val" or "test".

    Returns:
        Tuple[float, float]: auroc and accuracy result
    """
    model = train_pipeline._model
    model.eval()
    device = train_pipeline._device
    if limit_batches is not None:
        limit_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if limit_batches is None
        else itertools.islice(iterator, limit_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    auroc = metrics.AUROC(compute_on_step=False).to(device)
    accuracy = metrics.Accuracy(compute_on_step=False).to(device)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for _ in tqdm(iter(int, 1), desc=f"Evaluating {stage} set"):
        try:
            _loss, logits, labels = train_pipeline.progress(combined_iterator)
            preds = torch.sigmoid(logits)
            auroc(preds, labels)
            accuracy(preds, labels)
        except StopIteration:
            break
    auroc_result = auroc.compute().item()
    accuracy_result = accuracy.compute().item()
    if dist.get_rank() == 0:
        print(f"AUROC over {stage} set: {auroc_result}.")
        print(f"Accuracy over {stage} set: {accuracy_result}.")
    return auroc_result, accuracy_result


def _train(
    train_pipeline: TrainPipelineSparseDist,
    iterator: Iterator[Batch],
    next_iterator: Iterator[Batch],
    within_epoch_val_dataloader: DataLoader,
    epoch: int,
    epochs: int,
    change_lr: bool,
    lr_change_point: float,
    lr_after_change_point: float,
    validation_freq_within_epoch: Optional[int],
    limit_train_batches: Optional[int],
    limit_val_batches: Optional[int],
    prof=None,
    num_batches=0
) -> None:
    """
    Trains model for 1 epoch. Helper function for train_val_test.

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        iterator (Iterator[Batch]): Iterator used for training batches.
        next_iterator (Iterator[Batch]): Iterator used for validation batches
            in between epochs. Used to queue up the next TRAIN_PIPELINE_STAGES - 1
            batches before train_val_test switches to validation mode. This is done
            so that when validation starts, the first output train_pipeline generates
            an output for is the 1st validation batch (as opposed to a buffered train
            batch).
        within_epoch_val_dataloader (DataLoader): Dataloader to create iterators for
            validation within an epoch. This is only used if
            validation_freq_within_epoch is specified.
        epoch (int): Which epoch the model is being trained on.
        epochs (int): Number of epochs to train.
        change_lr (bool): Whether learning rate should be changed part way through
            training.
        lr_change_point (float): The point through training at which learning rate
            should change to the value set by lr_after_change_point.
            Applied only if change_lr is set to True.
        lr_after_change_point (float): Learning rate after change point in first epoch.
            Applied only if change_lr is set to True.
        validation_freq_within_epoch (Optional[int]): Frequency at which validation
            will be run within an epoch.
        limit_train_batches (Optional[int]): Number of train batches.
        limit_val_batches (Optional[int]): Number of validation batches.



    Returns:
        None.
    """
    train_pipeline._model.train()

    # For the first epoch, train_pipeline has no buffered batches, but for all other
    # epochs, train_pipeline will have TRAIN_PIPELINE_STAGES - 1 from iterator already
    # present in its buffer.
    if limit_train_batches is not None and epoch > 0:
        limit_train_batches -= TRAIN_PIPELINE_STAGES - 1

    # Because TrainPipelineSparseDist buffer batches internally, we load in
    # TRAIN_PIPELINE_STAGES - 1 batches from the next_iterator into the buffers so that
    # when train_val_test switches to the next phase, train_pipeline will start
    # producing results for the TRAIN_PIPELINE_STAGES - 1 buffered batches (as opposed
    # to the last TRAIN_PIPELINE_STAGES - 1 batches from iterator).
    combined_iterator = itertools.chain(
        iterator
        if limit_train_batches is None
        else itertools.islice(iterator, limit_train_batches),
        itertools.islice(next_iterator, TRAIN_PIPELINE_STAGES - 1),
    )
    samples_per_trainer = TOTAL_TRAINING_SAMPLES / dist.get_world_size() * epochs

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            train_pipeline.progress(combined_iterator)
            if it == (num_batches - 3):
                print(f"{get_mem_info('Training:  ')}")
            prof.step()
            if change_lr and (
                (it * (epoch + 1) / samples_per_trainer) > lr_change_point
            ):  # progress made through the epoch
                print(f"Changing learning rate to: {lr_after_change_point}")
                optimizer = train_pipeline._optimizer
                lr = lr_after_change_point
                for g in optimizer.param_groups:
                    g["lr"] = lr

            if validation_freq_within_epoch and it % validation_freq_within_epoch == 0:
                _evaluate(
                    limit_val_batches,
                    train_pipeline,
                    iter(within_epoch_val_dataloader),
                    iterator,
                    "val",
                )
                train_pipeline._model.train()
        except StopIteration:
            break


def train_val_test(
    args: argparse.Namespace,
    train_pipeline: TrainPipelineSparseDist,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    test_dataloader: DataLoader,
) -> TrainValTestResults:
    """
    Train/validation/test loop. Contains customized logic to ensure each dataloader's
    batches are used for the correct designated purpose (train, val, test). This logic
    is necessary because TrainPipelineSparseDist buffers batches internally (so we
    avoid batches designated for one purpose like training getting buffered and used for
    another purpose like validation).

    Args:
        args (argparse.Namespace): parsed command line args.
        train_pipeline (TrainPipelineSparseDist): pipelined model.
        train_dataloader (DataLoader): DataLoader used for training.
        val_dataloader (DataLoader): DataLoader used for validation.
        test_dataloader (DataLoader): DataLoader used for testing.

    Returns:
        TrainValTestResults.
    """

    train_val_test_results = TrainValTestResults()

    train_iterator = iter(train_dataloader)
    test_iterator = iter(test_dataloader)
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=0, warmup=len(train_dataloader)-3, active=2, repeat=1
            ),
            on_trace_ready=trace_handler,
    ) as prof:
        for epoch in range(args.epochs):
            val_iterator = iter(val_dataloader)
            _train(
                train_pipeline,
                train_iterator,
                val_iterator,
                val_dataloader,
                epoch,
                args.epochs,
                args.change_lr,
                args.lr_change_point,
                args.lr_after_change_point,
                args.validation_freq_within_epoch,
                args.limit_train_batches,
                args.limit_val_batches,
                prof,
                len(train_dataloader)
            )
            train_iterator = iter(train_dataloader)
            val_next_iterator = (
                test_iterator if epoch == args.epochs - 1 else train_iterator
            )

            val_accuracy, val_auroc = _evaluate(
                args.limit_val_batches,
                train_pipeline,
                val_iterator,
                val_next_iterator,
                "val",
            )

            train_val_test_results.val_accuracies.append(val_accuracy)
            train_val_test_results.val_aurocs.append(val_auroc)
            # prof.step()

        test_accuracy, test_auroc = _evaluate(
            args.limit_test_batches,
            train_pipeline,
            test_iterator,
            iter(test_dataloader),
            "test",
        )
        train_val_test_results.test_accuracy = test_accuracy
        train_val_test_results.test_auroc = test_auroc

    return train_val_test_results


def main(argv: List[str]) -> None:
    """
    Trains, validates, and tests a Deep Learning Recommendation Model (DLRM)
    (https://arxiv.org/abs/1906.00091). The DLRM model contains both data parallel
    components (e.g. multi-layer perceptrons & interaction arch) and model parallel
    components (e.g. embedding tables). The DLRM model is pipelined so that dataloading,
    data-parallel to model-parallel comms, and forward/backward are overlapped. Can be
    run with either a random dataloader or an in-memory Criteo 1 TB click logs dataset
    (https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

    Args:
        argv (List[str]): command line args.

    Returns:
        None.
    """
    args = parse_args(argv)
    if args.kaggle:
        global TOTAL_TRAINING_SAMPLES
        TOTAL_TRAINING_SAMPLES = 45840617
        setattr(args, 'num_embeddings_per_feature', NUM_EMBEDDINGS_PER_FEATURE)

    rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        dist.init_process_group(backend=backend)

    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(
            map(int, args.num_embeddings_per_feature.split(","))
        )
        args.num_embeddings = None

    # TODO add CriteoIterDataPipe support and add random_dataloader arg
    train_dataloader = get_dataloader(args, backend, "train")
    val_dataloader = get_dataloader(args, backend, "val")
    test_dataloader = get_dataloader(args, backend, "test")

    if dist.get_rank() == 0:
        print(args)
        print(f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
              f"test batches: {len(test_dataloader)}")
    # Sets default limits for random dataloader iterations when left unspecified.
    if args.in_memory_binary_criteo_path is None:
        for stage in STAGES:
            attr = f"limit_{stage}_batches"
            if getattr(args, attr) is None:
                setattr(args, attr, 10)

    eb_configs = [
        EmbeddingBagConfig(
            name=f"t_{feature_name}",
            embedding_dim=args.embedding_dim,
            num_embeddings=none_throws(args.num_embeddings_per_feature)[feature_idx]
            if args.num_embeddings is None
            else args.num_embeddings,
            feature_names=[feature_name],
        )
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
    ]
    sharded_module_kwargs = {}
    if args.over_arch_layer_sizes is not None:
        sharded_module_kwargs["over_arch_layer_sizes"] = list(
            map(int, args.over_arch_layer_sizes.split(","))
        )

    train_model = DLRMTrain(
        embedding_bag_collection=EmbeddingBagCollection(
            tables=eb_configs, device=torch.device("meta")
        ),
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,
    )
    fused_params = {
        "learning_rate": args.learning_rate,
        "optimizer": OptimType.EXACT_ROWWISE_ADAGRAD if args.adagrad else OptimType.EXACT_SGD,
    }
    sharders = [
        EmbeddingBagCollectionSharder(fused_params=fused_params),
    ]

    print(f"{get_mem_info('After model init:  ')}")

    # Torchrec Planner
    env = ShardingEnv.from_process_group(dist.GroupMember.WORLD)
    topology = Topology(
        world_size=env.world_size,
        compute_device="cuda",
        hbm_cap=70*1024**3,  # GPU mem
        ddr_cap=300*1024*3,  # CPU mem
        intra_host_bw=1000*1024**3/1000,  # Device to Device bandwidth
        # inter_host_bw=CROSS_NODE_BANDWIDTH,  # Not used yet
        batch_size=args.batch_size
    )
    constraints = {
        f"t_{feature_name}": ParameterConstraints(
            sharding_types=[ShardingType.TABLE_WISE.value]  # if num_embeddings < 1e5 else ShardingType.ROW_WISE.value],
        ) for num_embeddings, feature_name in zip(args.num_embeddings_per_feature, DEFAULT_CAT_NAMES)
    }
    planner = EmbeddingShardingPlanner(
        topology=topology,
        constraints=constraints,
    )
    plan = planner.collective_plan(train_model, sharders, env.process_group)
    model = DistributedModelParallel(
        module=train_model,
        device=device,
        sharders=cast(List[ModuleSharder[nn.Module]], sharders),
        plan=plan
    )
    print(f"{get_mem_info('After model parallel:  ')}")

    if dist.get_rank() == 0:
        print(f"Plan: {model.plan}")

    def optimizer_with_params():
        if args.adagrad:
            return lambda params: torch.optim.Adagrad(params, lr=args.learning_rate)
        else:
            return lambda params: torch.optim.SGD(params, lr=args.learning_rate)

    dense_optimizer = KeyedOptimizerWrapper(
        dict(model.named_parameters()),  # why dense? what about spares?
        optimizer_with_params(),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    train_pipeline = TrainPipelineSparseDist(
        model,
        optimizer,
        device,
    )
    train_val_test(
        args, train_pipeline, train_dataloader, val_dataloader, test_dataloader
    )


if __name__ == "__main__":
    main(sys.argv[1:])
