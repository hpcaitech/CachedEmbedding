import pytest
from functools import partial

import numpy as np
import torch
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
from colossalai.utils.cuda import get_current_device

from baselines.modules.colossal_embedding import EmbeddingCollection
from recsys.utils.distributed_manager import DISTMGR
from recsys.utils.launch import launch


def run_embedding_collection(inputs, num_embeddings_per_feature, use_cpu):
    device = get_current_device()
    model_device = device if not use_cpu else torch.device('cpu')

    inputs = inputs.to(model_device)
    offsets = torch.tensor([0, *np.cumsum([6, 4])[:-1]], dtype=torch.int64).unsqueeze(0).to(model_device)

    model = EmbeddingCollection(num_embeddings_per_feature, 2).to(model_device)

    target = torch.nn.Embedding.from_pretrained(model.weight.detach(), freeze=False).to(device)

    model_output = model(inputs)
    target_output = target(inputs.to(device) + offsets.to(device))
    assert model_output.device.type == target_output.device.type
    assert torch.allclose(model_output, target_output)

    grad = torch.rand_like(model_output)
    model_output.backward(grad)
    target_output.backward(grad)
    target_weight_grad = target.weight.grad.detach()
    if use_cpu:
        target_weight_grad = target_weight_grad.cpu()
    assert torch.allclose(model.weight.grad.detach(), target_weight_grad)


def run_dist(rank, world_size, port, use_cpu):
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    rank = DISTMGR.get_rank()

    num_embeddings_per_feature = [6, 4]

    torch.manual_seed(rank + 42)
    f1 = torch.randint(num_embeddings_per_feature[0], (2, 1))
    f2 = torch.randint(num_embeddings_per_feature[1], (2, 1))
    idx = torch.cat([f1, f2], dim=1)

    run_embedding_collection(idx, num_embeddings_per_feature, use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_embedding(world_size, use_cpu):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), use_cpu=use_cpu)
    torch.multiprocessing.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_embedding(4, use_cpu=True)
