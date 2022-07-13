import pytest
from functools import partial
import numpy as np
import torch
import torch.multiprocessing as mp

from recsys import launch, disable_existing_loggers
from recsys import DISTMGR
from recsys.testing import rerun_if_address_is_in_use, free_port
from recsys.modules.embeddings import CachedEmbeddingBag, ParallelCachedEmbeddingBag, CacheReplacePolicy

NUM_EMBEDDINGS, EMBEDDING_DIM = 100, 8
BATCH_SIZE = 8


def synthesize_sparse_feature(
    batch_size,
    num_embed,
    device,
):
    indices_in_batch = batch_size * 2
    indices = torch.randint(low=0, high=num_embed, size=(indices_in_batch,), device=device, dtype=torch.long)
    offsets = torch.from_numpy(
        np.array([
            0, *np.sort(np.random.randint(low=0, high=indices_in_batch, size=(indices_in_batch - 1,))), indices_in_batch
        ])).to(device).long()
    return indices, offsets


def run_cached_embedding_bag(cache_replace_policy):
    device = torch.device('cuda', 0)
    ref_model = torch.nn.EmbeddingBag(NUM_EMBEDDINGS,
                                      EMBEDDING_DIM,
                                      device=device,
                                      mode='mean',
                                      include_last_offset=True)
    model = CachedEmbeddingBag.from_pretrained(ref_model.weight.detach().cpu(),
                                               cache_sets=BATCH_SIZE * 2,
                                               cache_replace_policy=cache_replace_policy,
                                               mode='mean',
                                               include_last_offset=True,
                                               freeze=False).to(device)
    assert model.weight.device.type == 'cpu'
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for _ in range(3):
        indices, offsets = synthesize_sparse_feature(BATCH_SIZE, NUM_EMBEDDINGS, device)
        res = model(indices, offsets)
        ref_res = ref_model(indices, offsets)
        assert torch.allclose(res, ref_res), f"model result: {res}, reference: {ref_res}"

        grad = torch.rand_like(res)
        # comparing gradient here is nontrivial
        res.backward(grad)
        ref_res.backward(grad)
        optimizer.step()
        optimizer.zero_grad()

        ref_optimizer.step()
        ref_optimizer.zero_grad()

    model.flush_cache_()
    model_weight = model.weight.detach().to(device)
    ref_weight = ref_model.weight.detach()
    assert torch.allclose(model_weight, ref_weight)


def gather_tensor(tensor):
    gather_list = []
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    group = DISTMGR.get_group()
    torch.distributed.gather(tensor, gather_list, dst=0, group=group)
    return gather_list


def parallel_cached_embedding_bag(cache_replace_policy):
    device = torch.device("cuda", torch.cuda.current_device())
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()

    # indivisible
    weight = torch.randn(NUM_EMBEDDINGS, EMBEDDING_DIM, dtype=torch.float)
    model = ParallelCachedEmbeddingBag.from_pretrained(weight.detach(),
                                                       cache_sets=BATCH_SIZE * 2,
                                                       cache_replace_policy=cache_replace_policy,
                                                       mode='mean',
                                                       include_last_offset=True,
                                                       freeze=False).to(device)
    assert model.weight.device.type == 'cpu'
    weight_in_rank = torch.tensor_split(weight, world_size, 1)[rank]
    assert torch.allclose(weight_in_rank, model.weight.detach())

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    if rank == 0:
        ref_model = torch.nn.EmbeddingBag.from_pretrained(weight.cuda(),
                                                          mode='mean',
                                                          include_last_offset=True,
                                                          freeze=False)
        ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    # sync RNG states
    DISTMGR.set_seed(1234)
    for i in range(4):
        indices, offsets = synthesize_sparse_feature(BATCH_SIZE, NUM_EMBEDDINGS, device)
        res = model(indices, offsets)

        grad = torch.rand(BATCH_SIZE * 2, EMBEDDING_DIM, dtype=res.dtype, device=res.device)
        grad_in_rank = torch.tensor_split(grad, world_size, 0)[rank]
        res.backward(grad_in_rank)

        optimizer.step()
        optimizer.zero_grad()
        model.flush_cache_()

        res_list = gather_tensor(res.detach())
        weight_list = gather_tensor(model.weight.detach().cuda())
        # weight_str = '\n'.join([str((r, _w[:10])) for r, _w in enumerate(weight_list)])
        if rank == 0:
            ref_res = ref_model(indices, offsets)
            recover_res = torch.cat(res_list, dim=0)
            assert torch.allclose(ref_res.detach(), recover_res), f"model res: {recover_res}, ref res: {ref_res}"

            ref_res.backward(grad)
            ref_optimizer.step()
            ref_optimizer.zero_grad()
            # print(weight_str)
            recover_weight = torch.cat(weight_list, dim=1)
            # print(f"it {i}: model weight: {recover_weight[:10]}, ref weight: {ref_model.weight.detach()[:10]}")
            assert torch.allclose(recover_weight, ref_model.weight.detach())


def run_parallel_cached_embedding_bag(rank, world_size, port, cache_replace_policy):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    parallel_cached_embedding_bag(cache_replace_policy)


@pytest.mark.parametrize("cache_replace_policy", [CacheReplacePolicy.Hash, CacheReplacePolicy.LFU])
def test_cached_embedding_bag(cache_replace_policy,):
    run_func = partial(run_cached_embedding_bag, cache_replace_policy=cache_replace_policy)
    run_func()


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize("cache_replace_policy", [CacheReplacePolicy.Hash, CacheReplacePolicy.LFU])
@rerun_if_address_is_in_use()
def test_parallel_cached_embedding_bag(world_size, cache_replace_policy):
    run_func = partial(run_parallel_cached_embedding_bag,
                       world_size=world_size,
                       cache_replace_policy=cache_replace_policy,
                       port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == "__main__":
    # test_cached_embedding_bag(CacheReplacePolicy.Hash)
    test_parallel_cached_embedding_bag(4, CacheReplacePolicy.LFU)
