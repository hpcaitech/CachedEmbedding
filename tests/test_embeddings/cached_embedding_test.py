import pytest
from functools import partial
import numpy as np
import torch
import torch.multiprocessing as mp

from colossalai.utils import free_port
from colossalai.testing import rerun_if_address_is_in_use

from recsys import launch, disable_existing_loggers
from recsys import DISTMGR
from recsys.modules.embeddings import CachedEmbeddingBag, CacheReplacePolicy

NUM_EMBEDDINGS, EMBEDDING_DIM = 100, 8
BATCH_SIZE = 8


def synthesize_sparse_feature(
    batch_size,
    num_embed,
    device,
):
    indices_in_batch = BATCH_SIZE * 2
    indices = torch.randint(low=0, high=NUM_EMBEDDINGS, size=(indices_in_batch,), device=device, dtype=torch.long)
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


@pytest.mark.parametrize("cache_replace_policy", [CacheReplacePolicy.Hash, CacheReplacePolicy.LFU])
def test_cached_embedding_bag(cache_replace_policy,):
    run_func = partial(run_cached_embedding_bag, cache_replace_policy=cache_replace_policy)
    run_func()


if __name__ == "__main__":
    test_cached_embedding_bag(CacheReplacePolicy.Hash)
