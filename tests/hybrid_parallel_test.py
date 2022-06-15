import pytest
from functools import partial
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from recsys import DISTMGR, launch, disable_existing_loggers, ParallelMode
from recsys.modules.embeddings import ColumnParallelEmbeddingBag
from recsys.modules.functional import split_forward_gather_backward

BATCH_SIZE = 4


class Net_1(torch.nn.Module):

    def __init__(self, num_embeddings, embedding_dim, is_partial=True, is_dist=False):
        """
        is_partial:
            - True:     only linear is wrapped by DDP
            - False:    hybrid parallel, embed is tensor parallelized while linear is data parallelized
        is_dist:
            - True:     embedding vectors are split along the batch size dimension
            - False:    vanilla sequential model
        """
        super(Net_1, self).__init__()

        self.embed = torch.nn.EmbeddingBag(num_embeddings, embedding_dim, include_last_offset=True) if is_partial \
            else ColumnParallelEmbeddingBag(num_embeddings, embedding_dim, include_last_offset=True)
        self.linear = torch.nn.Linear(embedding_dim, 5)
        self.is_dist = is_dist

    def forward(self, sparse_features, offsets):
        embeddings = self.embed(sparse_features, offsets)
        if self.is_dist:
            embeddings = split_forward_gather_backward(embeddings, ParallelMode.DEFAULT, 0)
        return self.linear(embeddings)


def hybrid_ddp(use_cpu):
    """
    Embedding TP + Linear DP
    """
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    group = DISTMGR.get_group() if not use_cpu else DISTMGR.get_cpu_group()
    device = torch.device('cpu') if use_cpu else torch.device('cuda', torch.cuda.current_device())


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
    model = Net_1(num_embeddings, embedding_dim, is_partial=True, is_dist=True)
    # The method comment and torchrec's DistributedModelParallel use the names derived from model.named_parameters()
    # But, it seems that the references also work.
    DDP._set_params_and_buffers_to_ignore_for_model(module=model, params_and_buffers_to_ignore=[model.embed.weight])
    # device_ids must be None when using DDP on CPU
    model = DDP(model.to(device),
                device_ids=None if use_cpu else [rank],
                process_group=group,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True)

    if rank == 0:
        torch_model = Net_1(num_embeddings, embedding_dim, is_partial=True, is_dist=False).to(device)
        with torch.no_grad():
            torch_model.embed.weight.copy_(model.module.embed.weight.detach())
            torch_model.linear.weight.copy_(model.module.linear.weight.detach())
            torch_model.linear.bias.copy_(model.module.linear.bias.detach())

        assert torch.allclose(torch_model.embed.weight.detach(), model.module.embed.weight.detach())
        assert torch.allclose(torch_model.linear.weight.detach(), model.module.linear.weight.detach())

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

    print(f"rank: {rank}, model linear weight: {model.module.linear.weight}")

    if rank == 0:
        torch_outputs = torch_model(sparse_features, offsets)
        target_output = torch.tensor_split(torch_outputs.detach(), world_size, 0)[rank]
        assert torch.allclose(outputs, target_output)

        torch_outputs.backward(global_grad)
        torch.allclose(model.module.linear.weight.grad.detach(), torch_model.linear.weight.grad.detach())

        print(f"torch linear weight: {torch_model.linear.weight}")
        torch.allclose(model.module.linear.bias.grad.detach(), torch_model.linear.bias.grad.detach())
        torch.allclose(model.module.embed.weight.grad.detach(), torch_model.embed.weight.grad.detach())


def run_partial_ddp(rank, world_size, port, use_cpu):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    partial_ddp(use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [True])
@rerun_if_address_is_in_use()
def test_partial_ddp(world_size, use_cpu):
    run_func = partial(run_partial_ddp, world_size=world_size, use_cpu=use_cpu, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    test_partial_ddp(4, True)
