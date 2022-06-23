import pytest
from functools import partial
import torch
import torch.multiprocessing as mp

import colossalai
from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils.cuda import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.context.parallel_mode import ParallelMode
from colossalai.core import global_context as gpc
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.tensor import ParallelAction, ComputePattern, distspec, DistSpecManager, TensorSpec


def run_embedding_bag(use_cpu):
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    group = gpc.get_group(ParallelMode.PARALLEL_1D)

    num_embed, embed_dim = 4 * world_size, 2 * world_size
    device = get_current_device()
    with ColoInitContext(device=device):
        model = torch.nn.EmbeddingBag(num_embed, embed_dim, sparse=True, include_last_offset=True)

    ref_model = torch.nn.EmbeddingBag.from_pretrained(model.weight.detach(),
                                                      freeze=False,
                                                      sparse=True,
                                                      include_last_offset=True)

    spec = TensorSpec(distspec.shard(group, [-1], [world_size]), ParallelAction(ComputePattern.TP1D, gather_out=False))
    with DistSpecManager.no_grad():
        model.weight.set_spec(spec)

    # check init weight
    ref_weight = torch.tensor_split(ref_model.weight.detach(), world_size, 1)[rank]
    assert torch.allclose(ref_weight, model.weight.detach())

    model_device = device
    if use_cpu:
        model_device = torch.device('cpu')
        model.to(model_device)
        gloo_group = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
        model.weight.spec.dist_spec.process_group = gloo_group

    inputs = torch.randint(low=0, high=num_embed, size=(10,), device=model_device)
    offsets = torch.tensor([0, 4, 4, 8, 10], dtype=torch.long, device=model_device)

    model_res = model(inputs, offsets)
    ref_res = ref_model(inputs.to(device), offsets.to(device))

    # check results before communication
    if use_cpu:
        model_res = model_res.cuda()
        model_res.spec.dist_spec.process_group = group

    ref_res_before_comm = torch.tensor_split(ref_res.detach(), world_size, dim=1)[rank]
    assert torch.allclose(model_res.detach(), ref_res_before_comm)

    print(f"rank: {rank}, model res: {model_res.shape} spec: {model_res.spec}")
    model_res = model_res.view(4, 1, -1)
    # check results after communication
    model_res = model_res.convert_to_dist_spec(distspec.shard(group, [0], [world_size])).squeeze(1)
    print(f"rank: {rank}, after convert model res: {model_res.shape}, spec: {model_res.spec}")
    ref_res_after_comm = torch.tensor_split(ref_res.detach(), world_size, dim=0)[rank]
    assert torch.allclose(model_res.detach(), ref_res_after_comm)

    full_grad = torch.rand_like(ref_res)
    ref_res.backward(full_grad)
    grad_in_rank = torch.tensor_split(full_grad.detach(), world_size, 0)[rank]
    model_res.backward(grad_in_rank)

    # check grad
    ref_model_grad = torch.tensor_split(ref_model.weight.grad.detach().to_dense(), world_size, 1)[rank]
    assert torch.allclose(model.weight.grad.detach().to_dense().cuda(), ref_model_grad)


def run_dist(rank, world_size, port, use_cpu):
    colossalai.logging.disable_existing_loggers()

    config = dict(parallel=dict(tensor=dict(mode="1d", size=world_size),))
    colossalai.launch(config=config,
                      rank=rank,
                      world_size=world_size,
                      host='localhost',
                      port=port,
                      backend='nccl',
                      verbose=False)

    run_embedding_bag(use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_embedding_bag(world_size, use_cpu):
    run_func = partial(
        run_dist,
        world_size=world_size,
        port=free_port(),
        use_cpu=use_cpu,
    )

    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_embedding_bag(4, True)
