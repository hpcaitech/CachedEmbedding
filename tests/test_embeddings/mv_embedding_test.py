from functools import partial

import pytest
import torch 
import torch.nn as nn
import torch.multiprocessing as mp

from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from recsys import DISTMGR, launch, ParallelMode
from recsys import disable_existing_loggers
from recsys.modules.embeddings import LoadBalanceManager, BlockEmbeddingBag, ParallelMixVocabEmbeddingBag

from common import EMBEDDING_DIM, FIELD_DIMS, BATCH_SIZE, REDUCE_OPS, check_equal
from recsys.modules.functional import reduce_forward


def _print_rank_0(msg):
    i = DISTMGR.get_rank()
    if i == 0:
        print(msg)

def check_block_embedding():
    # embed = nn.EmbeddingBag(sum([FIELD_DIMS[i] for i in group]), block_dim)
    # embed = embed.to(dtype).to(device)

    # linear = nn.Linear(block_dim, EMBEDDING_DIM)
    # linear = linear.to(dtype).to(device)
    pass

def check_mv_embedding():
    device = get_current_device()
    dtype = torch.float32
    world_size = DISTMGR.get_world_size()

    lbmgr = LoadBalanceManager(FIELD_DIMS, world_size, EMBEDDING_DIM)

    rank = DISTMGR.get_rank()
    
    group = lbmgr.groups[rank]
    block_dim = lbmgr.emb_dims[rank]
    comm_func = reduce_forward # need all_reduce

    blk_embed = BlockEmbeddingBag(
                    sum([FIELD_DIMS[i] for i in group]),
                    block_dim,
                    EMBEDDING_DIM)

    blk_embed = blk_embed.to(dtype).to(device)

    test_embed = ParallelMixVocabEmbeddingBag.from_pretrained(
                    blk_embed=blk_embed, 
                    lbmgr=lbmgr,
                    comm_func=comm_func,
                    mode=REDUCE_OPS)

    test_embed = test_embed.to(dtype).to(device)

    A_shape = (BATCH_SIZE, len(FIELD_DIMS))
    A_master = torch.randint(min(FIELD_DIMS), A_shape, device=device)
    
    torch.distributed.broadcast(A_master, src=0)

    A_parallel = lbmgr.shard_tensor(A_master, rank)
    A_output_parallel = blk_embed(A_parallel)
    
    A_output_gather = comm_func(
                    A_output_parallel, 
                    ParallelMode.DEFAULT, 
                    reduce_op=REDUCE_OPS)
    
    A_output_gather_clone = A_output_gather.clone()

    A_output_gather_clone /= world_size

    A = A_master.clone()
    out = test_embed(A)

    check_equal(out.detach(), A_output_gather_clone.detach())
    _print_rank_0('embed forward: pass')

    grad_shape = A_output_gather.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    A_output_gather.backward(grad_master)

    blk_weights = blk_embed.get_weights()
    test_weights = test_embed.get_weights()
    
    for i in range(len(blk_weights)):
        check_equal(blk_weights[i].grad.detach(), test_weights[i].grad.detach())
    
    _print_rank_0('embed backward: pass')


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    
    check_mv_embedding()
    
    DISTMGR.destroy()
    torch.cuda.empty_cache()


@pytest.mark.parametrize('world_size', [1,4])
@rerun_if_address_is_in_use()
def test_layer(world_size):
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_layer(4)
