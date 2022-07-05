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

def check_block_embeddingbag():
    dtype = torch.float32
    
    example_field_idx = [0,2]
    example_fields = [FIELD_DIMS[i] for i in example_field_idx]
    example_block_dim = EMBEDDING_DIM // 2

    # torch modules
    embed = nn.EmbeddingBag(
                sum(example_fields), 
                example_block_dim,
                mode=REDUCE_OPS)
    embed = embed.to(dtype)

    linear = nn.Linear(
                example_block_dim, 
                EMBEDDING_DIM,
                bias=False)
    linear = linear.to(dtype)

    # block embedding bag
    blk_embed = BlockEmbeddingBag.from_pretrained(
                    weights=[embed.weight.clone().detach(),linear.weight.clone().detach()],
                    base_embedding_dim=EMBEDDING_DIM,
                    freeze=False,
                    mode=REDUCE_OPS)
    blk_embed = blk_embed.to(dtype)

    # initiate input tensor
    A_shape = (BATCH_SIZE, len(example_fields))
    A_master = torch.randint(min(example_fields), A_shape)
    
    # forward pass
    A = A_master.clone()
    torch_output = linear(embed(A))

    A_master = A_master.clone()
    blk_embed_output = blk_embed(A_master)

    check_equal(torch_output.detach(), blk_embed_output.detach())
    _print_rank_0('embed forward: pass')

    # backward pass
    grad_shape = torch_output.shape
    grad_master = torch.randn(grad_shape, dtype=dtype)

    grad = grad_master.clone()
    torch_output.backward(grad)

    grad_master = grad_master.clone()
    blk_embed_output.backward(grad_master)
    
    blk_embed_ws = blk_embed.get_weights(detach=False)
    
    check_equal(blk_embed_ws[0].grad, embed.weight.grad)        
    check_equal(blk_embed_ws[1].grad, linear.weight.grad)
    _print_rank_0('embed backward: pass')

def check_mv_embeddingbag():
    device = get_current_device()
    dtype = torch.float32
    world_size = DISTMGR.get_world_size()

    lbmgr = LoadBalanceManager(FIELD_DIMS, world_size, EMBEDDING_DIM)

    rank = DISTMGR.get_rank()
    
    group = lbmgr.get_group(rank)
    block_dim = lbmgr.get_block_dim(rank)
    comm_func = reduce_forward # need all_reduce

    blk_embed = BlockEmbeddingBag(
                    sum([FIELD_DIMS[i] for i in group]),
                    block_dim,
                    EMBEDDING_DIM)

    blk_embed = blk_embed.to(dtype).to(device)

    test_embed = ParallelMixVocabEmbeddingBag.from_pretrained(
                    blk_embed=blk_embed, 
                    lbmgr=lbmgr,
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
    
    if REDUCE_OPS == 'mean':
        A_output_gather = A_output_gather / world_size

    A = A_master.clone()
    test_out = test_embed(A)

    check_equal(test_out.detach(), A_output_gather.detach())
    _print_rank_0('embed forward: pass')

    grad_shape = A_output_gather.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    
    grad = grad_master.clone()
    A_output_gather.backward(grad)

    grad_master = grad_master.clone()
    test_out.backward(grad_master)

    blk_weights = blk_embed.get_weights(detach=False)
    test_weights = test_embed.get_weights(detach=False)
    
    for (w1,w2) in zip(blk_weights, test_weights):
        if w1 is None or w2 is None:
            assert w1 is None and w2 is None
        else:
            check_equal(w1.grad, w2.grad)
 
    _print_rank_0('embed backward: pass')

def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    
    # check_block_embeddingbag()
    check_mv_embeddingbag()
    
    DISTMGR.destroy()
    torch.cuda.empty_cache()

@pytest.mark.parametrize('world_size', [1,4])
@rerun_if_address_is_in_use()
def test_layer(world_size):
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_layer(4)
