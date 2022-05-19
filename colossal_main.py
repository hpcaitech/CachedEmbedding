import torch

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
from colossalai.core import global_context as gpc

from data.colossal_dataloader import get_dataloader, STAGES
from modules.dlrm_train import DLRMTrain
from dlrm_main import train_val_test


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
        "--validation_freq_within_epoch",
        type=int,
        default=None,
        help="Frequency at which validation will be run within an epoch.",
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

    # For compatibility with train_val_test
    for stage in STAGES:
        attr = f"limit_{stage}_batches"
        setattr(args, attr, None)
    return args


def main():
    args = parse_args()

    colossalai.logging.disable_existing_loggers()
    colossalai.launch_from_torch(config=args.config_path, verbose=False)

    logger = colossalai.logging.get_dist_logger()
    logger.info(f"launch rank {gpc.get_global_rank()}, Done")

    logger.info('Build data loader', ranks=[0])

    # TODO: check IterDataPipe dataloader
    train_dataloader = get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, "val")
    test_dataloader = get_dataloader(args, "test")

    logger.info(f"training batches: {len(train_dataloader)}, val batches: {len(val_dataloader)}, "
                f"test batches: {len(test_dataloader)}")

    logger.info('Build model', ranks=[0])
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
    train_model = DLRMTrain(
        embedding_bag_collection=ebc,
        dense_in_features=len(DEFAULT_INT_NAMES),
        dense_arch_layer_sizes=list(map(int, args.dense_arch_layer_sizes.split(","))),
        over_arch_layer_sizes=list(map(int, args.over_arch_layer_sizes.split(","))),
        dense_device=device,
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
        device=device,
        sharders=sharders,
    )

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


if __name__ == '__main__':
    main()
