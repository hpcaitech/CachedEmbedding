import pytest
from functools import partial
import torch
import torch.multiprocessing as mp

from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use

from recsys import launch, disable_existing_loggers
from recsys import DISTMGR as dist_manager
from recsys.modules.embeddings import VocabParallelEmbedding, ColumnParallelEmbeddingBag


def embedding(use_cpu, padding_idx):
    rank = dist_manager.get_rank()
    # assert get_current_device().index == rank  # not equal
    world_size = dist_manager.get_world_size()
    device = torch.device('cpu', rank) if use_cpu else torch.device('cuda', torch.cuda.current_device())

    torch_model = torch.nn.Embedding(16, 2, padding_idx=padding_idx).to(device)
    weight = torch_model.weight.detach().requires_grad_(True)
    model = VocabParallelEmbedding(16, 2, padding_idx=padding_idx, _weight=weight)
    model.to(device)
    assert model.weight.device.type == device.type
    assert model.weight.shape[0] == weight.shape[0] // world_size

    torch_model_chunk = torch.split(weight, weight.shape[0] // world_size, 0)[rank]
    assert torch.allclose(torch_model_chunk.detach(), model.weight.detach())

    data = torch.randint(0, 16, size=(2, 2), device=device)
    print(f"Rank: {rank}, data: {data}")

    x = model(data)
    torch_x = torch_model(data)
    torch_model_weight_chunk = torch.split(torch_model.weight.detach(), torch_model.weight.shape[0] // world_size,
                                           0)[rank]
    print(f"Rank: {rank}, model output: {x}, ref output: {torch_x}")
    print(f"Rank: {rank}, model weight: {model.weight}, ref weight: {torch_model_weight_chunk}")
    assert torch.allclose(model.weight.detach(), torch_model_weight_chunk)
    assert torch.allclose(x, torch_x)

    grad = torch.rand_like(x)
    x.backward(grad)
    torch_x.backward(grad)
    torch_grad_chunk = torch.split(torch_model.weight.grad, weight.shape[0] // world_size, 0)[rank]
    assert torch.allclose(model.weight.grad, torch_grad_chunk)


def embedding_bag(use_cpu, padding_idx, reduction_op, embedding_dim):
    rank = dist_manager.get_rank()
    world_size = dist_manager.get_world_size()
    device = torch.device('cpu', rank) if use_cpu else torch.device('cuda', torch.cuda.current_device())

    num_embeddings, embedding_dim = 16, embedding_dim
    torch_model = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        mode=reduction_op).to(device)
    weight = torch_model.weight.detach().requires_grad_(True)
    model = ColumnParallelEmbeddingBag(num_embeddings,
                                       embedding_dim,
                                       padding_idx=padding_idx,
                                       mode=reduction_op,
                                       _weight=weight).to(device)
    # Check device type
    assert model.weight.device.type == device.type

    # Check weight size
    chunk_size = embedding_dim // world_size
    if rank < embedding_dim % world_size:
        model_weight_chunk_size = chunk_size + 1
    else:
        model_weight_chunk_size = chunk_size
    assert model.weight.shape[1] == model_weight_chunk_size

    # Check weight
    torch_model_weight = torch.tensor_split(torch_model.weight.detach(), world_size, dim=1)[rank]
    assert torch.allclose(torch_model_weight, model.weight.detach())

    # 1D inputs
    inputs = torch.randint(low=0, high=num_embeddings, size=(5,), device=device)
    offsets = torch.tensor([0, 4], dtype=torch.long, device=device)

    torch_res = torch_model(inputs, offsets)
    model_res = model(inputs, offsets)
    print(f"rank: {rank}, torch res: {torch_res}, model res: {model_res}")
    assert torch.allclose(torch_res.detach(), model_res.detach())

    grad = torch.rand_like(torch_res)
    torch_res.backward(grad)
    model_res.backward(grad)

    torch_model_weight_grad_chunk = torch.tensor_split(torch_model.weight.grad.detach(), world_size, dim=1)[rank]
    assert torch.allclose(torch_model_weight_grad_chunk, model.weight.grad.detach())

    torch_model.zero_grad()
    model.zero_grad()

    # 2D inputs
    inputs = torch.randint(low=0, high=num_embeddings, size=(5, 3), device=device)
    if padding_idx is not None:
        inputs[2, -2:] = padding_idx

    torch_res = torch_model(inputs)
    model_res = model(inputs)
    print(f"rank: {rank}, torch res: {torch_res}, model res: {model_res}")
    assert torch.allclose(torch_res.detach(), model_res.detach())

    grad = torch.rand_like(torch_res)
    torch_res.backward(grad)
    model_res.backward(grad)
    torch_model_weight_grad_chunk = torch.tensor_split(torch_model.weight.grad.detach(), world_size, dim=1)[rank]
    assert torch.allclose(torch_model_weight_grad_chunk, model.weight.grad.detach())


def run_embedding(rank, world_size, port, use_cpu, padding_idx):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    embedding(use_cpu, padding_idx)


def run_embedding_bag(rank, world_size, port, use_cpu, padding_idx, reduction_op, embedding_dim):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')

    embedding_bag(use_cpu, padding_idx, reduction_op, embedding_dim)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@pytest.mark.parametrize('padding_idx', [None, 0])
@rerun_if_address_is_in_use()
def test_embedding(world_size, use_cpu, padding_idx):
    run_func = partial(run_embedding, world_size=world_size, port=free_port(), use_cpu=use_cpu, padding_idx=padding_idx)
    mp.spawn(run_func, nprocs=world_size)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@pytest.mark.parametrize('padding_idx', [None, 1])
@pytest.mark.parametrize('reduction_op', ['mean'])
@rerun_if_address_is_in_use()
def test_embedding_bag(world_size, use_cpu, padding_idx, reduction_op):
    embedding_dim = 7
    run_func = partial(run_embedding_bag,
                       world_size=world_size,
                       port=free_port(),
                       use_cpu=use_cpu,
                       padding_idx=padding_idx,
                       reduction_op=reduction_op,
                       embedding_dim=embedding_dim)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # test_embedding(4, True, None)
    test_embedding_bag(4, False, None, 'sum', 8)
