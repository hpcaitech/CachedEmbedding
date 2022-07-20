import pytest
import torch

from recsys.modules.embeddings import ChunkParamMgr, FreqAwareEmbeddingBag
from recsys.testing.utils import synthesize_1d_sparse_feature


# torch.set_printoptions(profile="full")


@pytest.mark.parametrize('chunk_size', [1, 3, 4, 11])
def test_uneven_weight(chunk_size):
    weight = torch.randn(11, 5)
    mgr = ChunkParamMgr(weight, chunk_size, 10)
    assert mgr.cpu_weight.shape[0] % chunk_size == 0
    

def test_chunkmgr_admit():
    model = torch.nn.EmbeddingBag(10000, 128)
    # 10 chunks, 5 in cuda
    mgr = ChunkParamMgr(model.weight, 1000, 5)
    assert mgr.cuda_chunk_num == 5

    mgr._admit(1)
    assert mgr.cached_chunk_table[0] == (1, 0)
    mgr._admit(8)
    assert mgr.cached_chunk_table[1] == (8, 1000)

    # now 3 chunk is available
    assert mgr.cuda_available_chunk_num() == 3

    mgr._evict()
    assert mgr.cuda_available_chunk_num() == 4

    mgr._prepare_chunks_on_cuda([9, 6, 5])
    mgr._prepare_chunks_on_cuda([3, 4, 5])
    print(mgr.cached_chunk_table)
    mgr.print_comm_stats()

    mgr.flush()
    assert mgr.cuda_available_chunk_num() == 5


def test_freq_aware_embed():
    NUM_EMBEDDINGS, EMBEDDING_DIM = 48, 8
    BATCH_SIZE = 8

    device = torch.device('cuda', 0)
    model = FreqAwareEmbeddingBag(
        NUM_EMBEDDINGS,
        EMBEDDING_DIM,
        mode='mean',
        include_last_offset=True,
    ).to(device)
    model._preprocess(chunk_size=10, cuda_chunk_num=BATCH_SIZE * 2, ids_freq_mapping=None)

    assert model.weight.shape[0] == NUM_EMBEDDINGS
    ref_model = torch.nn.EmbeddingBag.from_pretrained(model.weight.detach().to(device),
                                                      mode='mean',
                                                      include_last_offset=True,
                                                      freeze=False)

    assert torch.allclose(ref_model.weight.detach(), model.weight.detach().to(device))

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # ref_optimizer = torch.optim.SGD(ref_model.parameters(), lr=1e-3)

    for i in range(3):
        indices, offsets = synthesize_1d_sparse_feature(BATCH_SIZE, NUM_EMBEDDINGS, device)
        res = model(indices, offsets)
        ref_res = ref_model(indices, offsets)
        assert torch.allclose(res, ref_res), f"model result: {res}, reference: {ref_res}"

        grad = torch.rand_like(res)
        # comparing gradient here is nontrivial
        res.backward(grad)
        ref_res.backward(grad)
        # optimizer.step()
        # optimizer.zero_grad()

        # ref_optimizer.step()
        # ref_optimizer.zero_grad()

    model.chunk_weight_mgr.flush()
    model_weight = model.weight.detach().to(device)
    ref_weight = ref_model.weight.detach()
    assert torch.allclose(model_weight, ref_weight), \
        f"model weight: {model_weight[10:18, :8]}, reference: {ref_weight[10:18, :8]}"


if __name__ == '__main__':
    # test_freq_aware_embed()
    test_freq_aware_embed()
