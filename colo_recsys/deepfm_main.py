import time
import datetime
import itertools
from tqdm import tqdm

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from colossalai.utils import get_dataloader
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
import torch
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import wandb
from sklearn.metrics import roc_auc_score

from colo_recsys.datasets import MovieLens20MDataset, AvazuDataset, CriteoDataset
from colo_recsys.models import DeepFactorizationMachine
from colo_recsys.utils.common import (
    # CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, 
    # get_mem_info, 
    trace_handler,
    count_parameters,
    compute_throughput,
    EarlyStopper,
)
from recsys.modules.engine import TrainPipelineBase

args = None

def parse_dfm_args():

    parser = colossalai.get_default_parser()

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--dataset', type=str, default='ml20m')
    parser.add_argument('--dataset_path', nargs='?', default=None)
    parser.add_argument('--cache_path', nargs='?', default='../../deepfm-colossal/.criteo')
    
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
    parser.add_argument('--model_dir', type=str, default='ckpts/')
    
    parser.add_argument('--model_path', type=str, default=None)

    parser.add_argument('--use_zero', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--use_clip', action='store_true')
    
    # Wandb
    parser.add_argument('--repeated_runs', type=int, default=1)
    parser.add_argument('--group', type=str, default='')

    parser.add_argument('--enable_qr', action='store_true')
    
    parser.add_argument('--clip', type=float, default=0)
    parser.add_argument("--bound", type=float, default=1)

    args = parser.parse_args()

    return args

def train(engine, data_loader, device, prof, epoch, log_interval=100):
    engine.train()
    total_loss = 0

    pipe = TrainPipelineBase(engine, device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    with compute_throughput(len(data_loader)*len(next(iter(data_loader)))) as ctp:
        for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
            try:
                losses, _output, _label = pipe.progress(iterator)
                # print(losses)
                prof.step()

                total_loss += losses

            except StopIteration:
                break
    
    throughput = ctp()

    ave_fwd_throughput=sum(pipe.ave_forward_throughput[1:]) / (len(pipe.ave_forward_throughput) -1)
    ave_bwd_throughput=sum(pipe.ave_backward_throughput[1:]) / (len(pipe.ave_backward_throughput) -1)

    print('ave_forward_throughput is {:.4f} (ms)'.format(ave_fwd_throughput))
    print('ave_backward_throughput is {:.4f} (ms)'.format(ave_bwd_throughput))
    print('total_throughput is {:.4f} (ms)'.format(throughput))

    return total_loss

def test(engine, data_loader, device, epoch=0):
    engine.eval()

    targets, predicts = list(), list()
    # auroc = metrics.AUROC(compute_on_step=False).to(device)
    # accuracy = metrics.Accuracy(compute_on_step=False).to(device)

    pipe = TrainPipelineBase(engine, device)
    iterator = iter(data_loader)

    # Infinite iterator instead of while-loop to leverage tqdm progress bar.
    for it in tqdm(itertools.count(), desc=f"Epoch {epoch}"):
        try:
            with torch.no_grad():
                _losses, output, label = pipe.progress(iterator)
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
    # build logger
    colossalai.logging.disable_existing_loggers()
    logger = get_dist_logger()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True, 
                record_shapes=True,
                on_trace_ready=trace_handler, 
        ) as prof:
        
        with record_function('Dataloader'):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if args.dataset in ['movielens20m','ml20m','movielens','ml']:
                dataset = MovieLens20MDataset(args.dataset_path)
            elif args.dataset == 'avazu':
                dataset = AvazuDataset(args.dataset_path)
            elif args.dataset == 'criteo':
                dataset = CriteoDataset(args.dataset_path, args.cache_path)
            
            train_length = int(len(dataset) * 0.8)
            valid_length = int(len(dataset) * 0.1)
            test_length = len(dataset) - train_length - valid_length
            train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
                dataset, (train_length, valid_length, test_length))
            
            train_data_loader = get_dataloader(train_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
            valid_data_loader = get_dataloader(valid_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)
            test_data_loader = get_dataloader(test_dataset, shuffle=True, batch_size=args.batch_size, pin_memory=True, num_workers=16)

        if args.use_zero:
            shard_strategy = TensorShardStrategy()
            with ZeroInitContext(target_device=device, shard_strategy=shard_strategy, shard_param=False):
                model = DeepFactorizationMachine(dataset.field_dims, \
                                args.embed_dim, eval(args.mlp_dims), args.dropout, args.enable_qr)
            model = ShardedModelV2(model, shard_strategy)
        else:
            with record_function('Model initialization'):
                model = DeepFactorizationMachine(dataset.field_dims, \
                                args.embed_dim, eval(args.mlp_dims), args.dropout, args.enable_qr)

        logger.info(count_parameters(model,'Number of parameters'), ranks=[0])
            
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        if args.use_zero:
            optimizer = ShardedOptimizerV2(model, optimizer, initial_scale=2**5)
        
        with record_function('Colossal initialize'):
            engine, train_dataloader, test_dataloader, lr_scheduler = colossalai.initialize(model,\
                                                                                optimizer, \
                                                                                criterion, \
                                                                                train_data_loader, \
                                                                                test_data_loader)

    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True, 
            record_shapes=True,
            schedule=schedule(wait=0, warmup=30, active=2, repeat=1),
            # on_trace_ready=trace_handler, 
            on_trace_ready=tensorboard_trace_handler('log/deepfm-dp2-dbstr-qr-cowclip'),
    ) as prof:
        t0 = time.time()
    
        early_stopper = EarlyStopper(verbose=True)
        
        for epoch_i in range(args.epoch):
            train_loss = train(engine, train_dataloader, device, prof, epoch_i)
            auc = test(engine, valid_data_loader, device, epoch_i)

            logger.info(
            f"Epoch {epoch_i} - train loss: {train_loss:.5}, auc: {auc:.5}",ranks=[0])

            if args.use_wandb:
                wandb.log({'AUC score': auc})

            early_stopper(auc)

            if early_stopper.early_stop:
                print("Early stopping")
                break

        t3 = time.time()
        print('overall training time:',t3-t0,'s')

        auc = test(engine, test_dataloader, device)
        print(f'test auc: {auc:.5}\n')


if __name__ == '__main__':
    # launch distributed environment
    args = parse_dfm_args()
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
