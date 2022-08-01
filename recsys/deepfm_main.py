import time
import datetime
import itertools
from tqdm import tqdm

import torch
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import wandb
from sklearn.metrics import roc_auc_score

from recsys.utils import get_default_parser
from recsys import (disable_existing_loggers, launch_from_torch, ParallelMode, DISTMGR as dist_manager, DISTLogger as
                    dist_logger)
from colo_recsys.datasets import CriteoDataset
from recsys.datasets import criteo
from recsys.models import DeepFactorizationMachine
from colo_recsys.utils import (
    count_parameters,
)
from recsys.modules.engine import TrainPipelineBase

args = None


def parse_dfm_args():
    parser = get_default_parser()

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

    # Dataset
    parser.add_argument('-t','--use_torchrec_dl', action='store_true')
    parser.add_argument("--kaggle", action='store_false')
    parser.add_argument('--dataset_path', nargs='?', default='/criteo/train/')
    parser.add_argument('--cache_path', nargs='?', default='/.criteo') #'../../deepfm-colossal/.criteo'
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

    # Scale
    parser.add_argument("--memory_fraction", type=float, default=None)
    parser.add_argument("--multiples", type=int, default=1)
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=10,
        help="number of train batches",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=int,
        default=10,
        help="number of validation batches",
    )
    parser.add_argument(
        "--limit_test_batches",
        type=int,
        default=10,
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
    parser.add_argument('--embed_dim', type=int, default=128,
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
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epoch.')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    
    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--repeated_runs', type=int, default=1)
    parser.add_argument('--group', type=str, default='')

    # Embed
    parser.add_argument('-q','--enable_qr', action='store_true')
    
    # Tensorboard
    parser.add_argument('--tboard_name', type=str, default='mvembed-4tp')
    
    args = parser.parse_args()
    
    if args.kaggle:
        setattr(args, 'num_embeddings_per_feature', criteo.KAGGLE_NUM_EMBEDDINGS_PER_FEATURE)
    if args.num_embeddings_per_feature is not None:
        args.num_embeddings_per_feature = list(map(lambda x:int(x)*args.multiples, args.num_embeddings_per_feature.split(",")))

    for stage in criteo.STAGES:
        attr = f"limit_{stage}_batches"
        if getattr(args, attr) is None:
            setattr(args, attr, 10)

    return args

def train(model, criterion, optimizer, data_loader, device, prof, epoch, log_interval=100):
    model.train()
    total_loss = 0
    pipe = TrainPipelineBase(model, criterion, optimizer, device=device)
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

def test(model, criterion, data_loader, device, epoch=0):
    model.eval()

    targets, predicts = list(), list()

    pipe = TrainPipelineBase(model, criterion, device=device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            with torch.no_grad():
                _, output, label = pipe.progress(iterator)
                targets.extend(label.tolist())
                predicts.extend(output.tolist())

        except StopIteration:
            print('iteration stopped')
            break

    return roc_auc_score(targets, predicts)

def main(args):
    curr_device = torch.device('cuda', torch.cuda.current_device())

    if args.use_torchrec_dl:
        train_data_loader = criteo.get_dataloader(args, 'train')
        valid_data_loader = criteo.get_dataloader(args, 'val')
        test_data_loader = criteo.get_dataloader(args, "test")
    else:
        train_data_loader = CriteoDataset(args,mode='train')
        valid_data_loader = CriteoDataset(args,mode='val')
        test_data_loader = CriteoDataset(args,mode='test')

    model = DeepFactorizationMachine(args.num_embeddings_per_feature, len(criteo.DEFAULT_INT_NAMES),\
                    args.embed_dim, eval(args.mlp), args.dropout, args.enable_qr).to(curr_device)

    dist_logger.info(count_parameters(model,f'[rank{dist_manager.get_rank()}]model'), ranks=[0])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, 
            record_shapes=True,
            schedule=schedule(wait=0, warmup=30, active=2, repeat=1),
            # on_trace_ready=tensorboard_trace_handler(f'log/{args.tboard_name}'),
    ) as prof:
        t0 = time.time()
        for epoch_i in range(args.epoch):
            train_loss = train(model, criterion, optimizer, train_data_loader, curr_device, prof, epoch_i)
            auc = test(model, criterion, valid_data_loader, curr_device)
            print('valid auc:',auc)

            dist_logger.info(
            f"Epoch {epoch_i} - train loss: {train_loss:.5}",ranks=[0])

        t3 = time.time()
        dist_logger.info(f'overall training time:{t3-t0}s',ranks=[0])
        auc = test(model, criterion, test_data_loader, curr_device)
        dist_logger.info(f'test auc: {auc:.5}\n',ranks=[0])


if __name__ == '__main__':
    # launch distributed environment
    args = parse_dfm_args()
    disable_existing_loggers()
    
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    launch_from_torch(backend='nccl', seed=args.seed)
    dist_manager.new_process_group(2, ParallelMode.TENSOR_PARALLEL)
    print(dist_manager.get_distributed_info())
    
    dist_logger.info(f'Number of embeddings per feature: {args.num_embeddings_per_feature}',ranks=[0])

    if args.use_wandb:
        run = wandb.init(project="deepfm-colossal", entity="jiatongg", group=args.group, name=f'run{args.repeated_runs}-{dist_manager.get_world_size(ParallelMode.DATA)}GPUs-{datetime.datetime.now().strftime("%x")}')
        wandb.config = {
            'batch_size': args.batch_size,
            'epochs': args.epoch,
            'parallel_mode': dist_manager.get_distributed_info(),
        }

    main(args)
    if args.use_wandb:
        wandb.finish()
        