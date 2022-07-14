import time
import datetime
import itertools
from tqdm import tqdm

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from colossalai.utils import get_current_device
import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from sklearn.metrics import roc_auc_score
import wandb

from colo_recsys.datasets.colossal_dataloader import get_dataloader
from colo_recsys.models import DeepFactorizationMachine
from colo_recsys.utils import (
    get_model_mem,
)
from colo_recsys.modules.engine import TrainPipelineBase
from recsys.datasets import criteo

args = None


def parse_dfm_args():    
    parser = colossalai.get_default_parser()

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

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
    
    parser.add_argument("--memory_fraction", type=float, default=None)
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
    parser.add_argument("--num_embeddings", type=int, default=10000)

    
    # Model setting
    parser.add_argument(
        "--num_embeddings_per_feature",
        type=str,
        default=None,
        help="Comma separated max_ind_size per sparse feature. The number of embeddings"
        " in each embedding table. 26 values are expected for the Criteo dataset.",
    )
    parser.add_argument('--embed_dim', type=int, default=1024,
                        help='User / entity Embedding size.')
    parser.add_argument(
        "--mlp",
        type=str,
        default="[128]",
        help="Comma separated layer sizes for dense arch.",
    )
    parser.add_argument('--dropout', nargs='?', default=0.2,
                        help='Dropout probability w.r.t. message dropout for bi-interaction layer and each hidden layer. 0: no dropout.')
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=1,
                        help='Number of epoch.')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--repeated_runs', type=int, default=1)
    parser.add_argument('--group', type=str, default='')

    # Tensorboard
    parser.add_argument('--tboard_name', type=str, default='mvembed-4tp')
    
    args = parser.parse_args()
    
    if args.kaggle:
        setattr(args, 'num_embeddings_per_feature', criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)
    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(map(lambda x:int(x), args.num_embeddings_per_feature.split(",")))

    for stage in criteo.STAGES:
        attr = f"limit_{stage}_batches"
        if getattr(args, attr) is None:
            setattr(args, attr, 100)

    return args

def train(model, data_loader, device, prof, epoch, log_interval=100):
    model.train()
    total_loss = 0
    pipe = TrainPipelineBase(model, device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            losses, _, _ = pipe.progress(iterator)
            prof.step()
            
            total_loss += losses
            if args.use_wandb:
                wandb.log({'loss':losses})
        except StopIteration:
            break
    
    return total_loss

def test(model, data_loader, device, epoch=0):
    model.eval()

    targets, predicts = list(), list()

    pipe = TrainPipelineBase(model, device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            with torch.no_grad():
                _, output, label = pipe.progress(iterator)
                targets.extend(label.tolist())
                predicts.extend(output.tolist())

        except StopIteration:
            break

    return roc_auc_score(targets, predicts)

def main(args):
    colossalai.logging.disable_existing_loggers()
    logger = get_dist_logger()
    
    curr_device = get_current_device()
    
    train_dataloader = get_dataloader(args, 'train')
    val_dataloader = get_dataloader(args, "val")
    test_dataloader = get_dataloader(args, "test")

    model = DeepFactorizationMachine(args.num_embeddings_per_feature, len(criteo.DEFAULT_INT_NAMES),\
                    args.embed_dim, eval(args.mlp), args.dropout).to(curr_device)

    logger.info(get_model_mem(model,f'[rank{gpc.get_global_rank()}]model'), ranks=[0])

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    engine, _, _, _ = colossalai.initialize(model,optimizer,criterion)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, 
            record_shapes=True,
            schedule=schedule(wait=0, warmup=30, active=2, repeat=1),
            on_trace_ready=tensorboard_trace_handler(f'log/{args.tboard_name}'),
    ) as prof:
        t0 = time.time()
    
        for epoch_i in range(args.epoch):
            train_loss = train(engine, train_dataloader, curr_device, prof, epoch_i)

            logger.info(
            f"Epoch {epoch_i} - train loss: {train_loss:.5}, auc: {auc:.5}",ranks=[0])

            if args.use_wandb:
                wandb.log({'AUC score': auc})

        t3 = time.time()
        logger.info(f'overall training time:{t3-t0}s',ranks=[0])

        auc = test(engine, test_dataloader, curr_device)
        logger.info(f'test auc: {auc:.5}\n',ranks=[0])


if __name__ == '__main__':
    # launch distributed environment
    args = parse_dfm_args()
    
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    colossalai.launch_from_torch(args.config, verbose=False)

    if args.use_wandb:
        run = wandb.init(project="deepfm-colossal", entity="jiatongg", group=args.group, name=f'run{args.repeated_runs}-{gpc.get_world_size(ParallelMode.GLOBAL)}GPUs-{datetime.datetime.now().strftime("%x")}')
        wandb.config = {
            'batch_size': args.batch_size,
            'epochs': args.epoch,
            'parallel_mode': gpc.config.parallel,
            'amp': hasattr(gpc.config, 'fp16')
        }

    main(args)

    if args.use_wandb:
        wandb.finish()
        