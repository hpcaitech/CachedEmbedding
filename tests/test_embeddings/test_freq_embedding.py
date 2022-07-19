from recsys.modules.embeddings import ChunkCUDAWeightMgr
import torch

def test_chunkmgr_admit():
    model = torch.nn.EmbeddingBag(10000, 128)
    # 10 chunks, 5 in cuda
    mgr = ChunkCUDAWeightMgr(model.weight, 1000, 5)
    assert mgr.cuda_chunk_num == 5

    mgr._admit(1)
    assert mgr.cached_chunk_table[0] == (1, 0)
    mgr._admit(8)
    assert mgr.cached_chunk_table[1] == (8, 1000)

    # now 3 chunk is available
    assert mgr.cuda_available_chunk_num() == 3

    mgr._evict()
    assert mgr.cuda_available_chunk_num() == 4

    mgr._prepare_cuda_chunks([9, 6, 5])
    mgr._prepare_cuda_chunks([3, 4, 5])
    print(mgr.cached_chunk_table)

if __name__ == '__main__':
    test_chunkmgr_admit()