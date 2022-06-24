import pytest
from functools import partial
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from recsys import DISTMGR, launch, disable_existing_loggers, ParallelMode
from recsys.modules.embeddings import ColumnParallelEmbeddingBag, FusedHybridParallelEmbeddingBag
from recsys.modules.functional import split_forward_gather_backward

BATCH_SIZE = 4


class Net_1(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, is_partial_ddp=True, is_dist=False):
        """
        is_partial_ddp:
            - True:     only linear is wrapped by DDP
            - False:    hybrid parallel, embed is tensor parallelized while linear is data parallelized
        is_dist:
            - True:     embedding vectors are split along the batch size dimension
            - False:    vanilla sequential model, enable it when build a target model for test
        """
        super(Net_1, self).__init__()

        self.embed = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, include_last_offset=True) if is_partial_ddp \
            else ColumnParallelEmbeddingBag(num_embeddings, embedding_dim, include_last_offset=True)
        self.linear = torch.nn.Linear(embedding_dim, 5)
        self.is_dist = is_dist

    def forward(self, sparse_features, offsets, send_shape=None):
        embeddings = self.embed(sparse_features, offsets)
        if send_shape is not None:
            embeddings = embeddings.view(*send_shape)

        if self.is_dist:
            embeddings = split_forward_gather_backward(embeddings, ParallelMode.DEFAULT, 0)
        return self.linear(embeddings)


class Net_2(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, rank, group):
        super(Net_2, self).__init__()
        self.embed = FusedHybridParallelEmbeddingBag(num_embeddings,
                                                     embedding_dim,
                                                     include_last_offset=True,
                                                     output_device_type='cuda').to('cpu')

        self.linear = DDP(torch.nn.Linear(embedding_dim, 5).cuda(), device_ids=[rank], process_group=group)

    def forward(self, sparse_features, offsets, send_shape):
        x = self.embed(sparse_features, offsets, send_shape=send_shape)
        return self.linear(x)


def collect_weight(tensor, rank, world_size):
    tensor = tensor.cpu()
    gather_list = []
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for i in range(world_size)]

    group = DISTMGR.get_cpu_group()
    torch.distributed.gather(tensor, gather_list, dst=0, group=group)

    return gather_list


def hybrid_parallel(use_cpu):
    """
    Embedding TP + Linear DP
    """
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    group = DISTMGR.get_group() if not use_cpu else DISTMGR.get_cpu_group()
    device = torch.device('cpu') if use_cpu else torch.device('cuda', torch.cuda.current_device())
    DISTMGR.set_seed(1234)
    torch.manual_seed(1234)

    num_embeddings, embedding_dim = world_size * 2, world_size * 2
    model = Net_1(num_embeddings, embedding_dim, is_partial_ddp=False, is_dist=True)
    ignore_params = [key for key, _ in model.named_parameters() if 'embed' in key]
    # print(f"Rank: {rank}, ignore param names: {ignore_params}")
    DDP._set_params_and_buffers_to_ignore_for_model(module=model, params_and_buffers_to_ignore=ignore_params)
    model = DDP(
        model.to(device),
        device_ids=None if use_cpu else [rank],
        process_group=group,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    # static_graph=True
    )

    embed_weight_list = collect_weight(model.module.embed.weight.detach(), rank, world_size)
    if rank == 0:
        global_embed_weight = torch.cat(embed_weight_list, dim=1)
        torch_model = Net_1(num_embeddings, embedding_dim, is_partial_ddp=True, is_dist=False).to(device)
        assert list(torch_model.embed.weight.shape) == list(global_embed_weight.shape)

        with torch.no_grad():
            torch_model.embed.weight.copy_(global_embed_weight.detach())
            torch_model.linear.weight.copy_(model.module.linear.weight.detach())
            torch_model.linear.bias.copy_(model.module.linear.bias.detach())

    DISTMGR.set_seed(4321)

    # Synthesize data for embedding bag:
    sparse_features = torch.randint(low=0, high=num_embeddings, size=(6,), dtype=torch.long, device=device)
    offsets = torch.tensor([0, 2, 3, 3, 6], dtype=torch.long, device=device)

    outputs = model(sparse_features, offsets)
    assert outputs.shape[0] == BATCH_SIZE // world_size

    global_grad = torch.rand(BATCH_SIZE, outputs.shape[1], dtype=outputs.dtype, device=outputs.device)
    grad = torch.tensor_split(global_grad, world_size, dim=0)[rank]
    outputs.backward(grad)
    print(f"Rank {rank}, model weight grad: {model.module.embed.weight.grad}")

    embed_grad_list = collect_weight(model.module.embed.weight.grad.detach(), rank, world_size)
    full_output_list = collect_weight(outputs.detach(), rank, world_size)
    if rank == 0:
        torch_outputs = torch_model(sparse_features, offsets)
        full_outputs = torch.cat(full_output_list, dim=0)
        assert torch.allclose(full_outputs.cpu(), torch_outputs.detach().cpu())

        torch_outputs.backward(global_grad)

        assert torch.allclose(model.module.linear.weight.grad.detach() * world_size,
                              torch_model.linear.weight.grad.detach())
        assert torch.allclose(model.module.linear.bias.grad.detach() * world_size,
                              torch_model.linear.bias.grad.detach())

        embed_grad = torch.cat(embed_grad_list, dim=1)
        print(f"model embed grad: {embed_grad}")
        print(f"target embed grad: {torch_model.embed.weight.grad}")
        assert torch.allclose(embed_grad.cpu(), torch_model.embed.weight.grad.detach().cpu())


def partial_ddp(use_cpu):
    """
    Only part of a model is wrapped by DDP.
    Here, the model's embed is shared across the group, while the linear module is trained by data parallelism
    """
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    group = DISTMGR.get_group() if not use_cpu else DISTMGR.get_cpu_group()
    device = torch.device('cpu') if use_cpu else torch.device('cuda', torch.cuda.current_device())

    num_embeddings, embedding_dim = world_size * 2, world_size * 4
    model = Net_1(num_embeddings, embedding_dim, is_partial_ddp=True, is_dist=True)
    # To make DDP ignore all-reducing for submodule, we must use the ``name`` instead of references to nn.Parameter
    ignore_params = [key for key, _ in model.named_parameters() if 'embed' in key]
    # print(f"Rank: {rank}, ignore param names: {ignore_params}")
    DDP._set_params_and_buffers_to_ignore_for_model(module=model, params_and_buffers_to_ignore=ignore_params)
    # device_ids must be None when using DDP on CPU
    model = DDP(
        model.to(device),
        device_ids=None if use_cpu else [rank],
        process_group=group,
        gradient_as_bucket_view=True,
        broadcast_buffers=False,
    # static_graph=True
    )

    linear_weight_list = collect_weight(model.module.linear.weight.detach(), rank, world_size)
    if rank == 0:
        for i in range(len(linear_weight_list) - 1):
            assert torch.allclose(linear_weight_list[i], linear_weight_list[i + 1])

        torch_model = Net_1(num_embeddings, embedding_dim, is_partial_ddp=True, is_dist=False).to(device)
        with torch.no_grad():
            torch_model.embed.weight.copy_(model.module.embed.weight.detach())
            torch_model.linear.weight.copy_(model.module.linear.weight.detach())
            torch_model.linear.bias.copy_(model.module.linear.bias.detach())

        assert torch.allclose(torch_model.embed.weight.detach(), model.module.embed.weight.detach())
        assert torch.allclose(torch_model.linear.weight.detach(), model.module.linear.weight.detach())

    # The randomly initialization of torch_model would affect the RNG states in rank 0, so we must reset it
    DISTMGR.set_seed(1234)
    torch.manual_seed(1234)
    # Synthesize data for embedding bag:
    # Note: all ranks synthesize the same data because DISTMGR has set the same random seed for each rank
    #
    #                       (sparse feature id dim)
    # (batch size dim)          [id1, id2]
    #                           [id3,]
    #                           []
    #                           [id4, id5, id6]
    sparse_features = torch.randint(low=0, high=num_embeddings, size=(6,), dtype=torch.long, device=device)
    offsets = torch.tensor([0, 2, 3, 3, 6], dtype=torch.long, device=device)

    outputs = model(sparse_features, offsets)
    assert outputs.shape[0] == BATCH_SIZE // world_size, f"outputs shape: {outputs.shape}"

    global_grad = torch.rand(BATCH_SIZE, outputs.shape[1], dtype=outputs.dtype, device=outputs.device)
    grad = torch.tensor_split(global_grad, world_size, dim=0)[rank]
    outputs.backward(grad)

    # print(f"rank:{rank}, global grad: {global_grad} grad: {grad}")
    # print(f"rank: {rank}, model linear grad: {model.module.linear.weight.grad.detach()}")
    if rank == 0:
        torch_outputs = torch_model(sparse_features, offsets)
        target_output = torch.tensor_split(torch_outputs.detach(), world_size, 0)[rank]
        assert torch.allclose(outputs, target_output)

        torch_outputs.backward(global_grad)
        assert torch.allclose(model.module.linear.weight.grad.detach() * world_size,
                              torch_model.linear.weight.grad.detach())
        assert torch.allclose(model.module.linear.bias.grad.detach() * world_size,
                              torch_model.linear.bias.grad.detach())
        assert torch.allclose(model.module.embed.weight.grad.detach(), torch_model.embed.weight.grad.detach())


def hybrid_device():
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    group = DISTMGR.get_group()

    num_embeddings, embedding_dim = world_size * 4, world_size * 2
    batch_size, feature_size = world_size * 2, 3
    model = Net_2(num_embeddings, embedding_dim, rank, group)
    if rank == 0:
        for n, v in model.named_parameters():
            print(f"name: {n}, shape: {v.shape}, device: {v.device}")
    embed_weight_list = collect_weight(model.embed.weight.detach(), rank, world_size)
    linear_weight_list = collect_weight(model.linear.module.weight.detach(), rank, world_size)
    linear_bias_list = collect_weight(model.linear.module.bias.detach(), rank, world_size)
    if rank == 0:
        embed_weight = torch.cat(embed_weight_list, dim=1).cuda()
        for i in range(world_size - 1):
            assert torch.allclose(linear_weight_list[i], linear_weight_list[i + 1])
            assert torch.allclose(linear_bias_list[i], linear_bias_list[i + 1])
        ref_model = Net_1(num_embeddings, embedding_dim).cuda()
        with torch.no_grad():
            ref_model.embed.weight.copy_(embed_weight)
            ref_model.linear.weight.copy_(linear_weight_list[0])
            ref_model.linear.bias.copy_(linear_bias_list[0])
    # Sync RNG states
    DISTMGR.set_seed(42)

    indices_in_batch = batch_size * feature_size
    data = torch.randint(low=0, high=num_embeddings, size=(indices_in_batch,))

    offsets = torch.from_numpy(
        np.array([
            0, *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))), indices_in_batch
        ]))
    # print(f"rank: {rank}, data: {data}, offset: {offsets}")

    # batch size / world size, feature size, output dim
    out = model(data, offsets, send_shape=(batch_size, feature_size, -1))
    assert list(out.shape) == [batch_size // world_size, feature_size, 5]
    grad = torch.rand_like(out)
    out.backward(grad)

    output_list = collect_weight(out.detach(), rank, world_size)
    grad_list = collect_weight(grad.detach(), rank, world_size)
    embed_grad_list = collect_weight(model.embed.weight.grad.detach(), rank, world_size)
    if rank == 0:
        # batch size, feature size, output dim
        ref_out = ref_model(data.cuda(), offsets.cuda(), send_shape=(batch_size, feature_size, -1))

        global_out = torch.cat(output_list, dim=0).cuda()
        assert torch.allclose(global_out, ref_out.detach())

        global_grad = torch.cat(grad_list, dim=0).cuda()
        ref_out.backward(global_grad)

        embed_grad = torch.cat(embed_grad_list, dim=1).cuda()
        assert torch.allclose(ref_model.embed.weight.grad.detach(), embed_grad)

        assert torch.allclose(model.linear.module.weight.grad.detach() * world_size,
                              ref_model.linear.weight.grad.detach())
        assert torch.allclose(model.linear.module.bias.grad.detach() * world_size, ref_model.linear.bias.grad.detach())


def run_hybrid_parallel(rank, world_size, port, use_cpu):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    hybrid_parallel(use_cpu)


def run_partial_ddp(rank, world_size, port, use_cpu):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    partial_ddp(use_cpu)


def run_hybrid_device(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    hybrid_device()


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_partial_ddp(world_size, use_cpu):
    run_func = partial(run_partial_ddp, world_size=world_size, use_cpu=use_cpu, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_hybrid_parallel(world_size, use_cpu):
    run_func = partial(run_hybrid_parallel, world_size=world_size, use_cpu=use_cpu, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


@pytest.mark.parametrize('world_size', [1, 4])
@rerun_if_address_is_in_use()
def test_hybrid_device(world_size):
    run_func = partial(run_hybrid_device, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # test_partial_ddp(4, True)
    # test_hybrid_parallel(2, False)
    test_hybrid_device(4)
