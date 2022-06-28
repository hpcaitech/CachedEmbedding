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

from common import EMBEDDING_DIM, FIELD_DIMS, BATCH_SIZE, check_equal
from recsys.modules.functional import reduce_forward


def _print_rank_0(msg):
    i = DISTMGR.get_rank()
    if i == 0:
        print(msg)

def check_md_embedding():
    device = get_current_device()
    dtype = torch.float32
    world_size = DISTMGR.get_world_size()

    lbmgr = LoadBalanceManager(FIELD_DIMS, world_size)

    rank = DISTMGR.get_rank()
    
    group = lbmgr.groups[rank]
    block_dim = lbmgr.emb_dims[rank]
    comm_func = reduce_forward # need all_reduce

    embed = nn.EmbeddingBag(sum([FIELD_DIMS[i] for i in group]), block_dim)
    embed = embed.to(dtype).to(device)
    
    linear = nn.Linear(block_dim, EMBEDDING_DIM)
    linear = linear.to(dtype).to(device)

    test_embed = ParallelMixVocabEmbeddingBag(FIELD_DIMS, EMBEDDING_DIM)
    test_embed = test_embed.to(dtype).to(device)

    test_embed.embed = BlockEmbeddingBag.from_pretrained([embed.weight,linear.weight], EMBEDDING_DIM)

    A_shape = (BATCH_SIZE, len(FIELD_DIMS))
    A_master = torch.randint(min(FIELD_DIMS), A_shape, device=device)
    
    torch.distributed.broadcast(A_master, src=0)

    A_parallel = lbmgr.mapping_rule(A_master, rank)
    A_output_parallel = linear(embed(A_parallel))
    A_output_gather = comm_func(A_output_parallel)

    A = A_master.clone()
    out = test_embed(A)

    check_equal(out, A_output_gather)
    _print_rank_0('embed forward: pass')

    grad_shape = A_output_gather.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    A_output_gather.backward(grad_master)

    embed_grad = embed.weight.grad
    linear_grad = linear.weight.grad
    
    check_equal(embed_grad, test_embed.embed.weight.grad)
    check_equal(linear_grad, test_embed.embed.linear.weight.grad)
    _print_rank_0('embed backward: pass')


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    
    check_md_embedding()
    
    DISTMGR.destroy()
    torch.cuda.empty_cache()


@pytest.mark.parametrize('world_size', [2,4])
@rerun_if_address_is_in_use()
def test_layer(world_size):
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_layer(2)
