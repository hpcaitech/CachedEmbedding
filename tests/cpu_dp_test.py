import pytest
from functools import partial

import numpy as np
import torch
import colossalai
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.core import global_context as gpc
from colossalai.utils.cuda import get_current_device

import sys
sys.path.append('..')
from modules.colossal_embedding import EmbeddingCollection


def run_cpu(inputs, num_embeddings_per_feature):
    device = get_current_device()
    offsets = torch.tensor([0, *np.cumsum([6, 4])[:-1]], dtype=torch.int64).unsqueeze(0)

    model = EmbeddingCollection(num_embeddings_per_feature, 2, torch.device('cpu'))

    target = torch.nn.Embedding.from_pretrained(
        model.weight.detach(),
        freeze=False
    ).to(device)

    model_output = model(inputs)
    target_output = target(inputs.to(device) + offsets.to(device))
    print(model_output)
    print(target_output)
    assert model_output.device.type == target_output.device.type
    assert torch.allclose(model_output, target_output)

    grad = torch.rand_like(model_output)
    model_output.backward(grad)
    target_output.backward(grad)
    assert torch.allclose(model.weight.grad, target.weight.grad.detach().cpu())


def run_gpu(inputs, num_embeddings_per_feature):
    device = get_current_device()
    offsets = torch.tensor([0, *np.cumsum([6, 4])[:-1]], dtype=torch.int64).unsqueeze(0).to(device)

    inputs = inputs.to(device)
    model = EmbeddingCollection(num_embeddings_per_feature, 2, device)
    target = torch.nn.Embedding.from_pretrained(
        model.weight.detach(),
        freeze=False
    ).to(device)

    model_output = model(inputs)
    target_output = target(inputs+offsets)
    assert model_output.device.type == target_output.device.type
    assert torch.allclose(model_output, target_output)

    grad = torch.rand_like(model_output)
    model_output.backward(grad)
    target_output.backward(grad)
    assert torch.allclose(model.weight.grad, target.weight.grad)
    print(model_output)
    print(target_output)


def run_dist(rank, world_size, port):
    colossalai.launch(config=dict(),
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl',
                      verbose=False)

    num_embeddings_per_feature = [6, 4]

    torch.manual_seed(gpc.get_global_rank()+42)
    f1 = torch.randint(num_embeddings_per_feature[0], (2, 1))
    f2 = torch.randint(num_embeddings_per_feature[1], (2, 1))
    idx = torch.cat([f1, f2], dim=1)

    run_cpu(idx, num_embeddings_per_feature)
    run_gpu(idx, num_embeddings_per_feature)


@pytest.mark.parametrize('world_size',  [1, 4])
@rerun_if_address_is_in_use()
def test_embedding(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_embedding(4)
