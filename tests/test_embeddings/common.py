import torch

EMBEDDING_DIM = 8
FIELD_DIMS = [1000, 987, 1874, 1000, 4000]
BATCH_SIZE = 8
REDUCE_OPS = 'max'

def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1)