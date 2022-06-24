from functools import partial

import pytest
import torch 
import torch.multiprocessing as mp

from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port, get_current_device
from recsys import DISTMGR, launch
from recsys import disable_existing_loggers
from recsys.modules.embeddings import ParallelQREmbedding, ColumnParallelEmbeddingBag

from common import EMBEDDING_DIM, FIELD_DIMS, BATCH_SIZE, check_equal


def _print_rank_0(msg):
    i = DISTMGR.get_rank()
    if i == 0:
        print(msg)


def check_qr_embedding():
    device = get_current_device()
    dtype = torch.float32
    
    num_buckets = torch.tensor(sum(FIELD_DIMS) // 50, device=device)
    embed = ParallelQREmbedding(EMBEDDING_DIM, num_buckets)
    embed = embed.to(dtype).to(device)

    q_embed_master = ColumnParallelEmbeddingBag(num_embeddings=num_buckets, 
                                  embedding_dim=EMBEDDING_DIM,)
    q_embed_master = q_embed_master.to(dtype).to(device)
    r_embed_master = ColumnParallelEmbeddingBag(num_embeddings=num_buckets, 
                                  embedding_dim=EMBEDDING_DIM,)
    r_embed_master = r_embed_master.to(dtype).to(device)

    weight_q_master = q_embed_master.weight.data
    torch.distributed.broadcast(weight_q_master, src=0)
    weight_r_master = r_embed_master.weight.data
    torch.distributed.broadcast(weight_r_master, src=0)
    
    embed.q_embeddings.weight.data.copy_(weight_q_master)
    embed.r_embeddings.weight.data.copy_(weight_r_master)

    A_shape = (BATCH_SIZE, len(FIELD_DIMS))
    A_master = torch.randint(min(FIELD_DIMS), A_shape, device=device)
    torch.distributed.broadcast(A_master, src=0)
    A = A_master.clone()
    out = embed(A)

    A_master = A_master.clone()
    q_ind = torch.div(A_master, num_buckets, rounding_mode='floor')
    r_ind = torch.remainder(A_master, num_buckets)

    Q_master = q_embed_master(q_ind)
    R_master = r_embed_master(r_ind)

    C_master = torch.sum(Q_master * R_master, dim=1)
    
    C = C_master.clone()

    check_equal(out, C)
    _print_rank_0('embed forward: pass')

    grad_shape = C_master.shape
    grad_master = torch.randn(grad_shape, dtype=dtype, device=device)
    torch.distributed.broadcast(grad_master, src=0)
    grad = grad_master.clone()
    out.backward(grad)
    grad_master = grad_master.clone()
    C_master.backward(grad_master)

    Q_grad = q_embed_master.weight.grad
    R_grad = r_embed_master.weight.grad
    
    check_equal(Q_grad, embed.q_embeddings.weight.grad)
    check_equal(R_grad, embed.r_embeddings.weight.grad)
    _print_rank_0('embed backward: pass')


def check_layer(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    check_qr_embedding()
    torch.cuda.empty_cache()


@pytest.mark.dist
@rerun_if_address_is_in_use()
def test_layer():
    world_size = 4
    run_func = partial(check_layer, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_layer()
