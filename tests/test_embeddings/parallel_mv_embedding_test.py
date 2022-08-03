from functools import partial

import pytest
import torch 
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from recsys import DISTMGR, launch, ParallelMode
from recsys import disable_existing_loggers
from recsys.modules.embeddings import (LoadBalanceManager, BlockEmbeddingBag, ParallelMixVocabEmbeddingBag,
                                       QREmbeddingBag)

from common import EMBEDDING_DIM, NUM_EMBEDDINGS_PER_FEATURE, BATCH_SIZE, check_equal
from recsys.modules.functional import reduce_forward

NUM_EMBEDDINGS_PER_FEATURE = [100,2020,3204,3434302,3023,45459,123,4566]


def _print_rank_0(*msg):
    i = DISTMGR.get_rank()
    if i == 0:
        print(*msg)
        
def test_all_reduce(mode):
    device = get_current_device()
    rank = DISTMGR.get_rank()
  
def check_multi_block_embeddingbag(mode):
    device = get_current_device()
    world_size = DISTMGR.get_world_size()
    rank = DISTMGR.get_rank()
    
    ## NOTE: for test consistency block dimensions for sharded groups are the same as original unsharded dimension
    # the whole embedding w/o sharding
    BLOCK_EMBED_DIM = EMBEDDING_DIM // 2
    embed_weight = torch.randn(size=(sum(NUM_EMBEDDINGS_PER_FEATURE),BLOCK_EMBED_DIM),
                               requires_grad=True,device=device)
    linear_weight = torch.randn(size=(EMBEDDING_DIM,BLOCK_EMBED_DIM),
                               requires_grad=True,device=device)
    weights = [embed_weight, linear_weight]
    if rank == 0:
        embed = BlockEmbeddingBag.from_pretrained(
                                weights,
                                EMBEDDING_DIM,
                                device=device,
                                mode=mode)
    
    # example random input
    _A = []
    for i in range(len(NUM_EMBEDDINGS_PER_FEATURE)):
        _A.append(torch.randint(0,NUM_EMBEDDINGS_PER_FEATURE[i],
                                               size=(BATCH_SIZE,)).unsqueeze(1))
    A = torch.cat(_A,dim=1).to(device)
    A_grad = torch.randn(size=(BATCH_SIZE,EMBEDDING_DIM)).to(device)

    ## table-wise sharding (not fair)
    lbmgr = LoadBalanceManager(
        NUM_EMBEDDINGS_PER_FEATURE,
        world_size,EMBEDDING_DIM,
        do_fair=True,device=device)
    
    shard_weights = [lbmgr.shard_weights(weights[0],rank).clone(),weights[1].clone()]
    num_embeddings_this_rank = lbmgr.get_num_embeddings_on_rank(rank)
    
    # test weight sharding correctness
    if rank != world_size - 1: # not last rank
        assert torch.allclose(shard_weights[0], 
                              weights[0][num_embeddings_this_rank*rank:
                                  num_embeddings_this_rank*(rank+1),:])
    else:
        assert torch.allclose(shard_weights[0], 
                              weights[0][num_embeddings_this_rank*rank:,:])
    
    shard_embed = BlockEmbeddingBag.from_pretrained(shard_weights,
                                                    EMBEDDING_DIM,
                                                    device=device,
                                                    mode=mode)
    
    # forward test
    if rank == 0:
        A_master = A.clone() + lbmgr.get_offsets(return_all=True)
        C = embed(A_master)
    shard_A = lbmgr.shard_tensor(A, rank)
    shard_C = shard_embed(shard_A)
    test_C = reduce_forward(shard_C, parallel_mode=ParallelMode.DEFAULT, reduce_op=mode)
    if rank == 0:
        print(C)
        print(test_C)
        assert torch.allclose(C, test_C, atol = 1e-2)
    
    # backward test
    if rank == 0:
        C.backward(A_grad.clone())
        test_C.backward(A_grad.clone())
        # embed layer
        embed_grad = lbmgr.shard_weights(embed.get_weights(False)[0].grad, rank)
        test_embed_grad = shard_embed.get_weights(False)[0].grad[:-1,:] # last one was padding idx
        assert torch.allclose(embed_grad,test_embed_grad)

def check_single_block_embeddingbag(mode):
    dtype = torch.float32
    
    example_field_idx = [0,2]
    example_fields = [NUM_EMBEDDINGS_PER_FEATURE[i] for i in example_field_idx]
    example_block_dim = EMBEDDING_DIM // 2

    # torch modules
    embed = nn.EmbeddingBag(
                sum(example_fields), 
                example_block_dim,
                mode=mode)
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
                    mode=mode)
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

def check_mv_embeddingbag(do_fair, mode):
    device = get_current_device()
    dtype = torch.float32
    world_size = DISTMGR.get_world_size()

    lbmgr = LoadBalanceManager(NUM_EMBEDDINGS_PER_FEATURE, world_size, \
                            EMBEDDING_DIM, device=device, do_fair=do_fair)

    rank = DISTMGR.get_rank()

    num_embeddings_on_rank = lbmgr.get_num_embeddings_on_rank(rank)
    block_dim = lbmgr.get_block_dim(rank)
    comm_func = reduce_forward # need all_reduce
    
    pretrain_embed = BlockEmbeddingBag(
                    num_embeddings_on_rank,
                    block_dim,
                    EMBEDDING_DIM,
                    device=device)
            
    test_embed = ParallelMixVocabEmbeddingBag.from_pretrained(
                    pretrain_embed=pretrain_embed, 
                    lbmgr=lbmgr,
                    mode=mode)

    test_embed = test_embed.to(dtype).to(device)
    
    rand_input = []
    
    for i in range(len(NUM_EMBEDDINGS_PER_FEATURE)):
        rand_input.append(torch.randint(0,NUM_EMBEDDINGS_PER_FEATURE[i],
                                               size=(BATCH_SIZE,)).unsqueeze(1))
        
    A_master = torch.cat(rand_input,dim=1).to(device)

    torch.distributed.broadcast(A_master, src=0)

    A_parallel = lbmgr.shard_tensor(A_master, rank)
    A_output_parallel = pretrain_embed(A_parallel)

    A_output_gather = comm_func(
                    A_output_parallel, 
                    ParallelMode.DEFAULT,
                    reduce_op=mode)

    if mode == 'mean':
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

    blk_weights = pretrain_embed.get_weights(detach=False)
    test_weights = test_embed.get_weights(detach=False)

    for (w1,w2) in zip(blk_weights, test_weights):
        if w1 is None or w2 is None:
            assert w1 is None and w2 is None
        else:
            check_equal(w1.grad, w2.grad)

    _print_rank_0('embed backward: pass')

def check_layer(rank, world_size, port, do_fair, mode):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    
    # check_single_block_embeddingbag(mode)
    check_multi_block_embeddingbag(mode)
    # check_mv_embeddingbag(do_fair, mode)
    
    DISTMGR.destroy()
    torch.cuda.empty_cache()

@pytest.mark.parametrize('do_fair', [True,False])
@pytest.mark.parametrize('world_size', [1,2,4])
@pytest.mark.parametrize('mode', ['sum','mean','max'])
@rerun_if_address_is_in_use()
def test_layer(world_size, do_fair, mode):
    run_func = partial(check_layer, world_size=world_size, port=free_port(), do_fair=do_fair, mode=mode)
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_layer(4, True, 'mean')
