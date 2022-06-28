from colossalai.utils import free_port, get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.tensor import TensorSpec, ComputePattern, ComputeSpec, DistSpecManager, distspec

from functools import partial
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode

from colossalai.nn.parallel.layers import init_colo_module
from colossalai.nn.parallel import ColoDDP

import colossalai
import torch
import torch.multiprocessing as mp
import numpy as np
import pytest

NUM_EMBEDDINGS = 16
EMBED_DIM = 4
OUTPUT_DIM = 5


def gather_tensor(tensor):
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    if world_size == 1:
        return [tensor]

    gather_list = []
    rank = gpc.get_local_rank(ParallelMode.PARALLEL_1D)
    group = gpc.get_group(ParallelMode.PARALLEL_1D) if tensor.device.type == 'cuda' \
        else gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.gather(tensor, gather_list, dst=0, group=group)
    return gather_list


class Net(torch.nn.Module):

    def __init__(self, num_embed, embed_dim, output_dim, use_cpu):
        super(Net, self).__init__()
        self.use_cpu = use_cpu
        self.embed = torch.nn.Embedding(num_embed, embed_dim, sparse=True)
        self.proj = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        group = gpc.get_group(ParallelMode.PARALLEL_1D)
        world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)

        x = self.embed(x)
        if self.use_cpu:
            # print(f"before .cuda: {x.spec.dist_spec}")
            x = x.cuda()
            x.tensor_spec.dist_spec.process_group = group
        #     print(f"after .cuda: {x.spec.dist_spec}")
        # print(f"Before: {x.shape}, {x.device}, dist spec: {x.spec.dist_spec}")
        x = x.convert_to_dist_spec(distspec.shard(group, [0], [world_size]))
        # print(f"After: {x.shape}, {x.device}, dist spec: {x.spec.dist_spec}")
        x = self.proj(x)
        return x


class RefNet(Net):

    def __init__(self, *args, **kwargs):
        super(RefNet, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = self.embed(x)
        return self.proj(x)


class EBNet(torch.nn.Module):

    def __init__(self, num_embed, embed_dim, output_dim, use_cpu):
        super(EBNet, self).__init__()
        self.embed = torch.nn.EmbeddingBag(num_embed, embed_dim, sparse=True, include_last_offset=True)
        self.linear = torch.nn.Linear(embed_dim, output_dim)
        self.use_cpu = use_cpu

    def forward(self, inputs, offsets, send_shape=None):
        embeddings = self.embed(inputs, offsets)
        if send_shape is not None:
            embeddings = embeddings.view(*send_shape)
        return self.linear(embeddings)


class ParallelEBNet(EBNet):

    def __init__(self, parallel_mode, *args, **kwargs):
        super(ParallelEBNet, self).__init__(*args, **kwargs)
        self.parallel_mode = parallel_mode
        self.group = gpc.get_group(parallel_mode)
        self.world_size = gpc.get_world_size(parallel_mode)

    def forward(self, inputs, offsets, send_shape=None):
        embeddings = self.embed(inputs, offsets)

        if self.use_cpu:
            embeddings = embeddings.cuda()
            embeddings.tensor_spec.dist_spec.process_group = self.group

        if send_shape is not None:
            spec = embeddings.tensor_spec.dist_spec
            embeddings = embeddings.view_base(*send_shape)
            embeddings.tensor_spec.dist_spec = spec

        embeddings = embeddings.convert_to_dist_spec(distspec.shard(self.group, [0], [self.world_size]))
        return self.linear(embeddings)


def run_hybrid_device_embedding_bag(use_cpu):
    rank = gpc.get_global_rank()
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    group = gpc.get_group(ParallelMode.PARALLEL_1D)
    device = get_current_device()

    batch_size, feature_size = world_size * 2, 3

    with ColoInitContext(device=device):
        model = ParallelEBNet(ParallelMode.PARALLEL_1D, NUM_EMBEDDINGS, EMBED_DIM, OUTPUT_DIM, use_cpu)

    if rank == 0:
        ref_model = EBNet(NUM_EMBEDDINGS, EMBED_DIM, OUTPUT_DIM, use_cpu).cuda()
        with torch.no_grad():
            ref_model.embed.weight.copy_(model.embed.weight)
            ref_model.linear.weight.copy_(model.linear.weight)
            ref_model.linear.bias.copy_(model.linear.bias)

    origin_size = model.embed.weight.shape[1]

    compute_spec = ComputeSpec(ComputePattern.TP1D)
    compute_spec.output_replicate = False
    spec = TensorSpec(distspec.shard(group, [-1], [world_size]), compute_spec)
    with DistSpecManager.no_grad():
        model.embed.weight.set_tensor_spec(spec)

    if use_cpu:
        model.embed.to('cpu')
        if gpc.tensor_parallel_size > 1:
            gloo_group_tp = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
            model.embed.weight.tensor_spec.dist_spec.process_group = gloo_group_tp

    ignore_params = []
    for k, v in model.named_parameters():
        if 'embed' in k:
            ignore_params.append(v)
    ColoDDP.set_params_to_ignore(ignore_params)
    model = ColoDDP(model, group)

    # Init check
    assert model.module.embed.weight.shape[1] == origin_size // world_size
    embed_weight_list = gather_tensor(model.module.embed.weight.detach())
    if rank == 0:
        embed_weight = torch.cat(embed_weight_list, dim=1).cuda()
        assert torch.allclose(embed_weight, ref_model.embed.weight.detach())

    optimizer = torch.optim.SGD([{
        "params": model.module.embed.parameters(),
        "lr": 1e-3
    }, {
        "params": model.module.linear.parameters(),
        "lr": 1e-3 * world_size
    }])
    if rank == 0:
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for i in range(3):
        torch.manual_seed(1234 + i * 3)
        # Synth data
        indices_in_batch = batch_size * feature_size
        data = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(indices_in_batch,))
        offsets = torch.from_numpy(
            np.array([
                0, *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))),
                indices_in_batch
            ]))

        if not use_cpu:
            data = data.cuda()
            offsets = offsets.cuda()

        outputs = model(data, offsets, send_shape=(batch_size, feature_size, -1))
        assert list(outputs.shape) == [batch_size // world_size, feature_size, OUTPUT_DIM]

        loss = outputs.sum()
        model.backward(loss)

        outputs_list = gather_tensor(outputs.detach())
        embed_grad_list = gather_tensor(model.module.embed.weight.grad.detach().to_dense())
        if rank == 0:
            ref_out = ref_model(data.cuda(), offsets.cuda(), send_shape=(batch_size, feature_size, -1))
            global_output = torch.cat(outputs_list, dim=0).cuda()
            assert torch.allclose(global_output, ref_out.detach())

            ref_loss = ref_out.sum()
            ref_loss.backward()

            embed_grad = torch.cat(embed_grad_list, dim=1).cuda()
            assert torch.allclose(ref_model.embed.weight.grad.detach().to_dense(), embed_grad)
            assert torch.allclose(model.module.linear.weight.grad.detach() * world_size,
                                  ref_model.linear.weight.grad.detach())
            assert torch.allclose(model.module.linear.bias.grad.detach() * world_size,
                                  ref_model.linear.bias.grad.detach())

            ref_optimizer.step()
            ref_optimizer.zero_grad()

        optimizer.step()
        optimizer.zero_grad()

        embed_weight_list = gather_tensor(model.module.embed.weight.detach())
        if rank == 0:
            embed_weight = torch.cat(embed_weight_list, dim=1).cuda()
            assert torch.allclose(embed_weight, ref_model.embed.weight.detach())
            assert torch.allclose(model.module.linear.weight.detach(), ref_model.linear.weight.detach())
            assert torch.allclose(model.module.linear.bias.detach(), ref_model.linear.bias.detach())


def run_hybrid_device(use_cpu):
    rank = gpc.get_global_rank()
    world_size = gpc.get_world_size(ParallelMode.PARALLEL_1D)
    device = get_current_device()
    with ColoInitContext(device=device):
        model = Net(NUM_EMBEDDINGS, EMBED_DIM, OUTPUT_DIM, use_cpu)

    if rank == 0:
        ref_model = RefNet(NUM_EMBEDDINGS, EMBED_DIM, OUTPUT_DIM, use_cpu).to(device)
        with torch.no_grad():
            ref_model.embed.weight.copy_(model.embed.weight)
            ref_model.proj.weight.copy_(model.proj.weight)
            ref_model.proj.bias.copy_(model.proj.bias)

    # sync RNG states
    torch.manual_seed(42)

    org_size = model.embed.weight.size(1)
    print(f'Rank: {rank}, embedding size: {model.embed.weight.size()} | device: {model.embed.weight.device}')
    parallel_action = ComputeSpec(ComputePattern.TP1D)
    parallel_action.output_replicate = False
    init_colo_module(model.embed, parallel_action, recursive=True, mode='col')

    # use cpu gloo to handle embedding
    if use_cpu:
        model.embed.to('cpu')
        if gpc.tensor_parallel_size > 1:
            gloo_group_tp = gpc.get_cpu_group(ParallelMode.PARALLEL_1D)
            model.embed.weight.tensor_spec.dist_spec.process_group = gloo_group_tp

    ignore_params = []
    for k, v in model.named_parameters():
        if 'embed' in k:
            ignore_params.append(v)
    ColoDDP.set_params_to_ignore(ignore_params)
    group = gpc.get_group(ParallelMode.PARALLEL_1D)
    model = ColoDDP(model, group)

    if rank == 0:
        for name, param in model.named_parameters():
            print(f"Name: {name}, shape: {param.shape}, spec: {param.tensor_spec.dist_spec}")
    optimizer = torch.optim.SGD([{
        "params": model.module.embed.parameters(),
        "lr": 1e-3
    }, {
        "params": model.module.proj.parameters(),
        "lr": 1e-3 * world_size
    }])
    print(f'Rank: {rank}, new embedding size: {model.module.embed.weight.size()} | '
          f'new device: {model.module.embed.weight.device}')
    assert model.module.embed.weight.size(1) == org_size // gpc.get_world_size(ParallelMode.PARALLEL_1D)

    # Make sure the weight correctly initialized as a vanilla model
    embed_weight_list = gather_tensor(model.module.embed.weight.detach())
    if rank == 0:
        embed_weight = torch.cat(embed_weight_list, dim=1).cuda()
        assert torch.allclose(embed_weight, ref_model.embed.weight.detach())

    data = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(8,))
    if not use_cpu:
        data = data.cuda()
    print(f"Rank: {rank}, data: {data[:3]}")
    out = model(data)
    loss = out.sum()
    model.backward(loss)

    embed_grad_list = gather_tensor(model.module.embed.weight.grad.detach().to_dense())
    output_list = gather_tensor(out.detach())
    if rank == 0:
        embed_grad = torch.cat(embed_grad_list, dim=1).cuda()
        outputs = torch.cat(output_list, dim=0)

        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)
        # check output
        ref_out = ref_model(data.to(device))
        assert torch.allclose(outputs, ref_out.detach())

        # check gradient, note that dp module gradients are averaged by world size in DDP
        ref_out.sum().backward()
        assert torch.allclose(ref_model.embed.weight.grad.detach().to_dense(), embed_grad)
        assert torch.allclose(ref_model.proj.weight.grad.detach(),
                              model.module.proj.weight._saved_grad.detach() * world_size)
        assert torch.allclose(ref_model.proj.bias.grad.detach(),
                              model.module.proj.bias._saved_grad.detach() * world_size)

        ref_optimizer.step()

    optimizer.step()

    # checkout weight after update
    embed_weight_list = gather_tensor(model.module.embed.weight.detach())
    if rank == 0:
        embed_weight = torch.cat(embed_weight_list, dim=1).cuda()
        assert torch.allclose(embed_weight, ref_model.embed.weight.detach())
        assert torch.allclose(model.module.proj.weight.detach(), ref_model.proj.weight.detach())
        assert torch.allclose(model.module.proj.bias.detach(), ref_model.proj.bias.detach())


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
    # run_hybrid_device(use_cpu)
    run_hybrid_device_embedding_bag(use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
# Working for simulate the embedding(CPU DP+TP) -> nn(GPU DP+TP)
def test_hybrid_device(world_size, use_cpu):
    run_func = partial(run_dist, world_size=world_size, port=free_port(), use_cpu=use_cpu)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_hybrid_device(4, False)
