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
from colossalai.tensor import ComputeSpec, ComputePattern, distspec, DistSpecManager, TensorSpec


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
    compute_spec = ComputeSpec(ComputePattern.TP1D)
    compute_spec.output_replicate = False
    spec = TensorSpec(distspec.shard(group, [-1], [world_size]), compute_spec)
    with DistSpecManager.no_grad():
        model.weight.set_tensor_spec(spec)

    # check init weight
    ref_weight = torch.tensor_split(ref_model.weight.detach(), world_size, 1)[rank]
    assert torch.allclose(ref_weight, model.weight.detach())

    model_device = device
    if use_cpu:
        model_device = torch.device('cpu')
        model.to(model_device)
        gloo_group = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
        model.weight.tensor_spec.dist_spec.process_group = gloo_group

    inputs = torch.randint(low=0, high=num_embed, size=(10,), device=model_device)
    offsets = torch.tensor([0, 4, 4, 8, 10], dtype=torch.long, device=model_device)

    model_res = model(inputs, offsets)
    ref_res = ref_model(inputs.to(device), offsets.to(device))

    # check results before communication
    if use_cpu:
        model_res = model_res.cuda()
        model_res.tensor_spec.dist_spec.process_group = group

    ref_res_before_comm = torch.tensor_split(ref_res.detach(), world_size, dim=1)[rank]
    assert torch.allclose(model_res.detach(), ref_res_before_comm)
    shard_spec = model_res.tensor_spec.dist_spec

    if rank == 0:
        # (4, 2)
        print(f"rank: {rank}, model res after model fwd: {model_res.shape} spec: {model_res.tensor_spec.dist_spec}")

    model_res = model_res.view_base(4, 1, 2)
    if rank == 0:
        # (4, 1, 2)
        print(f"rank: {rank}, after view model res: {model_res.shape}, spec: {model_res.tensor_spec.dist_spec}")
    model_res.tensor_spec.dist_spec = shard_spec
    if rank == 0:
        # (4, 1, 2)
        print(f"rank: {rank}, after reset spec model res: {model_res.shape}, spec: {model_res.tensor_spec.dist_spec}")

    # reset shard spec
    model_res.tensor_spec.dist_spec.dims = [-1]
    model_res = model_res.to_replicate()
    if rank == 0:
        print(f"rank: {rank}, after to_replicate res: {model_res.shape}, spec: {model_res.tensor_spec.dist_spec}")
        print(f"rank: {rank}, ref_res {ref_res.shape}")
    assert torch.allclose(model_res.view(4, embed_dim), ref_res), f"{model_res} vs {ref_res}"

    full_grad = torch.rand_like(ref_res)
    ref_res.backward(full_grad)
    # grad_in_rank = torch.tensor_split(full_grad.detach(), world_size, 0)[rank]
    model_res.backward(full_grad.unsqueeze(1))

    # check grad
    ref_model_grad = torch.tensor_split(ref_model.weight.grad.detach().to_dense(), world_size, 1)[rank]
    a = model.weight.grad.detach().to_dense().cuda()
    b = ref_model_grad
    assert torch.allclose(a, b), f"{a} vs {b}"


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
    test_embedding_bag(4, False)
