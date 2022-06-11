import pytest
from functools import partial
import torch
import torch.multiprocessing as mp

from colossalai.utils import free_port, get_current_device
from colossalai.testing import rerun_if_address_is_in_use

from recsys import launch
from recsys import distributed_manager as dist_manager
from recsys.modules.embeddings import VocabParallelEmbedding


def run_embedding(use_cpu, padding_idx):
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

    torch_model_chunk = torch.split(weight, weight.shape[0]//world_size, 0)[rank]
    assert torch.allclose(torch_model_chunk.detach(), model.weight.detach())

    data = torch.randint(0, 16, size=(2, 2), device=device)
    print(f"Rank: {rank}, data: {data}")

    x = model(data)
    torch_x = torch_model(data)
    torch_model_weight_chunk = torch.split(torch_model.weight.detach(), torch_model.weight.shape[0]//world_size, 0)[rank]
    print(f"Rank: {rank}, model output: {x}, ref output: {torch_x}")
    print(f"Rank: {rank}, model weight: {model.weight}, ref weight: {torch_model_weight_chunk}")
    assert torch.allclose(model.weight.detach(), torch_model_weight_chunk)
    assert torch.allclose(x, torch_x)

    grad = torch.rand_like(x)
    x.backward(grad)
    torch_x.backward(grad)
    torch_grad_chunk = torch.split(torch_model.weight.grad, weight.shape[0]//world_size, 0)[rank]
    assert torch.allclose(model.weight.grad, torch_grad_chunk)


def run(rank, world_size, port, use_cpu, padding_idx):
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_embedding(use_cpu, padding_idx)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@pytest.mark.parametrize('padding_idx', [None, 0])
@rerun_if_address_is_in_use()
def test_embedding(world_size, use_cpu, padding_idx):
    run_func = partial(run, world_size=world_size, port=free_port(), use_cpu=use_cpu, padding_idx=padding_idx)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_embedding(4, True, None)
