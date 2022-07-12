import time
import datetime
import itertools
from tqdm import tqdm

import torch
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import wandb
from sklearn.metrics import roc_auc_score

from recsys.utils import get_default_parser
from recsys import (disable_existing_loggers, launch_from_torch, ParallelMode, DISTMGR as dist_manager, DISTLogger as
                    dist_logger)
from recsys.modules.dataloader import get_dataloader
from colo_recsys.datasets import CriteoDataset
from recsys.models import DeepFactorizationMachine
from colo_recsys.utils.common import (
    # CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, 
    count_parameters,
    EarlyStopper,
)
from recsys.modules.engine import TrainPipelineBase

args = None

def parse_dfm_args():
    parser = get_default_parser()

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

    # Dataset
    parser.add_argument('--dataset', type=str, default='criteo')
    parser.add_argument('--dataset_path', nargs='?', default=None)
    parser.add_argument('--cache_path', nargs='?', default='../../deepfm-colossal/.criteo')
    
    # Model setting
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='User / entity Embedding size.')
    parser.add_argument('--mlp_dims', nargs='?', default='[128]',
                        help='Output sizes of every hidden layer.')
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

    # Embed
    parser.add_argument('--enable_qr', action='store_true')
    
    # Visual
    parser.add_argument('--tboard_name', type=str, default='mvembed-2tp2dp')
    
    args = parser.parse_args()

    return args

def train(model, criterion, optimizer, data_loader, device, prof, epoch, log_interval=100):
    model.train()
    total_loss = 0

    pipe = TrainPipelineBase(model, criterion, optimizer, device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            losses, _, _ = pipe.progress(iterator)
            # print(losses)
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
    # auroc = metrics.AUROC(compute_on_step=False).to(device)
    # accuracy = metrics.Accuracy(compute_on_step=False).to(device)

    pipe = TrainPipelineBase(model, criterion, device=device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            with torch.no_grad():
                _, output, label = pipe.progress(iterator)
                targets.extend(label.tolist())
                predicts.extend(output.tolist())
            # auroc(output, label)
            # accuracy(output, label)

        except StopIteration:
            break

    # auroc_result = auroc.compute().item()
    # accuracy_result = accuracy.compute().item()

    return roc_auc_score(targets, predicts)

def main(args):
    device = torch.device('cuda', torch.cuda.current_device())
    if args.dataset == 'criteo':
        dataset = CriteoDataset(args.dataset_path, args.cache_path)
    else:
        print('dataset not supported')
        dataset = CriteoDataset(args.dataset_path, args.cache_path)
    
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    
    train_data_loader = get_dataloader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    valid_data_loader = get_dataloader(valid_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
    test_data_loader = get_dataloader(test_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)

    model = DeepFactorizationMachine(dataset.field_dims, \
                    args.embed_dim, eval(args.mlp_dims), args.dropout, args.enable_qr).to(device)

    dist_logger.info(count_parameters(model,'Number of parameters'), ranks=[0])

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, 
            record_shapes=True,
            schedule=schedule(wait=0, warmup=30, active=2, repeat=1),
            # on_trace_ready=trace_handler, 
            on_trace_ready=tensorboard_trace_handler(f'log/{args.tboard_name}'),
    ) as prof:
        t0 = time.time()
    
        early_stopper = EarlyStopper(verbose=True)
        
        for epoch_i in range(args.epoch):
            train_loss = train(model, criterion, optimizer, train_data_loader, device, prof, epoch_i)
            auc = test(model, criterion, valid_data_loader, device, epoch_i)

            dist_logger.info(
            f"Epoch {epoch_i} - train loss: {train_loss:.5}, auc: {auc:.5}",ranks=[0])

            if args.use_wandb:
                wandb.log({'AUC score': auc})

            early_stopper(auc)

            if early_stopper.early_stop:
                print("Early stopping")
                break

        t3 = time.time()
        print('overall training time:',t3-t0,'s')

        auc = test(model, criterion, test_data_loader, device)
        print(f'test auc: {auc:.5}\n')


if __name__ == '__main__':
    # launch distributed environment
    args = parse_dfm_args()
    disable_existing_loggers()
    
    launch_from_torch(backend='nccl', seed=args.seed)
    dist_manager.new_process_group(4, ParallelMode.TENSOR_PARALLEL)
    
    print(dist_manager.get_distributed_info())

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
        