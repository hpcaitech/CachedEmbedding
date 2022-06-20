import time
import datetime
import itertools
from tqdm import tqdm
from typing import Optional
import math

import colossalai
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context import ParallelMode
from colossalai.utils import get_dataloader
from colossalai.zero.init_ctx import ZeroInitContext
from colossalai.zero.shard_utils import TensorShardStrategy
from colossalai.zero.sharded_model import ShardedModelV2
from colossalai.zero.sharded_optim import ShardedOptimizerV2
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.utils.cuda import get_current_device
from colossalai.tensor import (
    ColoTensor, 
    TensorSpec, 
    ComputePattern, 
    ParallelAction, 
    DistSpecManager, 
    distspec
)
import torch
import torchmetrics as metrics
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler
import wandb
from sklearn.metrics import roc_auc_score

from datasets.criteo import MovieLens20MDataset, AvazuDataset, CriteoDataset
from models import DeepFactorizationMachine
from utils.common import (
    # CAT_FEATURE_COUNT, DEFAULT_CAT_NAMES, 
    # trace_handler,
    # get_mem_info, 
    count_parameters,
    compute_throughput,
    MultipleOptimizer,
    EarlyStopper,
)

args = None

def parse_dfm_args():

    parser = colossalai.get_default_parser()

    parser.add_argument('--seed', type=int, default=2019,
                        help='Random seed.')

    parser.add_argument('--dataset', type=str, default='ml20m')
    parser.add_argument('--dataset_path', nargs='?', default=None)
    parser.add_argument('--cache_path', nargs='?', default='../deepfm-colossal/.criteo')
    
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
    parser.add_argument('--epoch', type=int, default=1000,
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


def cow_clip(w, g, ratio=1, ids=None, cnts=None, min_w=0.03, const=False):

    if isinstance(g, dict):
        values = torch.tensor(g.values())
        clipnorm = torch.norm(torch.gather(w, g.indices), axis=-1)
    else:
        values = g
        if const:
            clipnorm = torch.tensor([min_w] * g.shape[0], requires_grad=False)
        else:
            clipnorm = torch.linalg.norm(w, axis=-1)
            # bound weight norm by min_w
            clipnorm = torch.max(clipnorm, clipnorm.new_tensor([min_w] * clipnorm.size(0), requires_grad=False))

    clip_t = ratio * clipnorm
    l2sum_row = torch.sum(values * values, axis=-1)
    pred = l2sum_row > 0
    l2sum_row_safe = torch.where(pred, l2sum_row, torch.ones_like(l2sum_row))
    l2norm_row = torch.sqrt(l2sum_row_safe)

    intermediate = values * clip_t.unsqueeze(-1)

    g_clip = intermediate / torch.maximum(l2norm_row, clip_t).unsqueeze(-1)

    return g_clip

def _to_device(batch, device: torch.device, non_blocking: bool):
    return batch.to(device=device, non_blocking=non_blocking)

def _wait_for_batch(batch, stream: Optional[torch.cuda.streams.Stream]) -> None:
    if stream is None:
        return
    torch.cuda.current_stream().wait_stream(stream)
    cur_stream = torch.cuda.current_stream()
    batch.record_stream(cur_stream)

class TrainPipelineBase:
    """
    This class runs training iterations using a pipeline of two stages, each as a CUDA
    stream, namely, the current (default) stream and `self._memcpy_stream`. For each
    iteration, `self._memcpy_stream` moves the input from host (CPU) memory to GPU
    memory, and the default stream runs forward, backward, and optimization.
    """

    def __init__(
        self,
        engine: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self._model = engine
        self._device = device
        self._memcpy_stream: Optional[torch.cuda.streams.Stream] = (
            torch.cuda.Stream() if device.type == "cuda" else None
        )
        self._cur_batch = None
        self._label = None
        self._connected = False
        self.ave_forward_throughput = []
        self.ave_backward_throughput = []

    def _connect(self, dataloader_iter) -> None:
        cur_batch, label = next(dataloader_iter)
        self._cur_batch = cur_batch
        self._label = label
        with torch.cuda.stream(self._memcpy_stream):
            self._cur_batch = _to_device(cur_batch, self._device, non_blocking=True)
            self._label = _to_device(label, self._device, non_blocking=True)

        self._connected = True

    def progress(self, dataloader_iter):

        global args

        if not self._connected:
            self._connect(dataloader_iter)

        # Fetch next batch
        with record_function("## next_batch ##"):
            next_batch, next_label = next(dataloader_iter)
        
        cur_batch = self._cur_batch
        label = self._label
        assert cur_batch is not None and label is not None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._model.zero_grad()

        with record_function("## wait_for_batch ##"):
            _wait_for_batch(cur_batch, self._memcpy_stream)
            _wait_for_batch(label, self._memcpy_stream)

        with record_function("## forward ##"):
            with compute_throughput(len(cur_batch)) as ctp:
                output = self._model(cur_batch)

        fwd_throughput = ctp()
        # print('forward_throughput is {:.4f}'.format(fwd_throughput))
        self.ave_forward_throughput.append(fwd_throughput)
            
        with record_function("## criterion ##"):
            losses = self._model.criterion(output, label.float())

        if args.use_wandb:
            wandb.log({'loss':losses})

        if self._model.training:
            with record_function("## backward ##"):
                with compute_throughput(len(cur_batch)) as ctp: 
                    self._model.backward(losses)
        
        bwd_throughput = ctp()
        # print('backward_throughput is {:.4f}'.format(bwd_throughput))
        self.ave_backward_throughput.append(bwd_throughput) 

        # Cowclip
        gradients = []
        trainable_vars = list(self._model.model.named_parameters())

        for param in trainable_vars:
            gradients.append(param[1].grad)


        embed_index = [
            i for i, x in enumerate(trainable_vars) if "embedding" in x[0]
        ]
        dense_index = [i for i in range(
            len(trainable_vars)) if i not in embed_index]
        
        embed_vars = [trainable_vars[i] for i in embed_index]
        dense_vars = [trainable_vars[i] for i in dense_index]
        embed_gradients = [gradients[i] for i in embed_index]
        dense_gradients = [gradients[i] for i in dense_index]

        # CowClip
        if args.use_clip:
            lower_bound = args.clip * math.sqrt(args.embed_dim) * args.bound
            embed_gradients_clipped = []
            for w, g in zip(embed_vars, embed_gradients):
                
                if 'embedding' not in w[0]:
                    embed_gradients_clipped.append(g)
                    continue

                g_clipped = cow_clip(w[1], torch.reshape(g, w[1].size()), ratio=args.clip,
                                            ids=None, cnts=None, min_w=lower_bound)

                embed_gradients_clipped.append(g_clipped)

            embed_gradients = embed_gradients_clipped

        gradients = embed_gradients + dense_gradients

        with torch.no_grad():
            index = embed_index+dense_index 
            for i,j in enumerate(index):
                trainable_vars[j][1].grad = gradients[i]

        # Copy the next batch to GPU
        # self._cur_batch = cur_batch = next_batch
        self._cur_batch = next_batch
        if self._model.training:
            # self._label = label = next_label
            self._label = next_label
        else:
            self._label = next_label
        with record_function("## copy_batch_to_gpu ##"):
            with torch.cuda.stream(self._memcpy_stream):
                self._cur_batch = _to_device(self._cur_batch, self._device, non_blocking=True)
                self._label = _to_device( self._label, self._device, non_blocking=True)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._model.step()

        return losses.item(), output, label

def train(engine, data_loader, device, logger, use_zero, prof, epoch, log_interval=100):
    engine.train()
    total_loss = 0

    pipe = TrainPipelineBase(engine, device)
    iterator = iter(data_loader)
    # combined_iterator = itertools.chain(iterator)

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

    # combined_iterator = itertools.chain(iterator)

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

    return  roc_auc_score(targets, predicts)

def main(args):
    # build logger
    colossalai.logging.disable_existing_loggers()
    logger = get_dist_logger()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                profile_memory=True, 
                record_shapes=True,
                # on_trace_ready=trace_handler, 
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
                # with ColoInitContext(device=get_current_device()):
                model = DeepFactorizationMachine(dataset.field_dims, \
                                args.embed_dim, eval(args.mlp_dims), args.dropout, args.enable_qr)

        # Mixed optimizer
        trainable_vars = list(model.named_parameters())

        embed_index = [
            i for i, x in enumerate(trainable_vars) if "embedding" in x[0]
        ]
        dense_index = [i for i in range(len(trainable_vars)) if i not in embed_index]
        
        embed_vars = [trainable_vars[i][1] for i in embed_index]
        dense_vars = [trainable_vars[i][1] for i in dense_index]

        optimizer = MultipleOptimizer(torch.optim.Adam(params=embed_vars, lr=args.lr, weight_decay=args.weight_decay), 
                                torch.optim.Adam(params=dense_vars, lr=args.lr, weight_decay=args.weight_decay))
        

        logger.info(count_parameters(model,'Number of parameters'), ranks=[0])
            
        # spec = TensorSpec(
        #     distspec.shard(gpc.get_group(ParallelMode.PARALLEL_1D), [0], [gpc.get_world_size(ParallelMode.PARALLEL_1D)]),
        #     ParallelAction(ComputePattern.TP1D))
        # with DistSpecManager.no_grad():
        #     # model.linear.weight.set_spec(spec)
        #     model.embedding.weight.set_spec(spec)
            # weights = model.mlp.weight
            # for weight in weights:
            #     weight.set_spec(spec)

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
        
        for epoch_i in range(gpc.config.NUM_EPOCHS):
            train_loss = train(engine, train_dataloader, device, logger, args.use_zero, prof, epoch_i)
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
