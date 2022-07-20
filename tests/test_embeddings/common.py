import torch

EMBEDDING_DIM = 8
NUM_EMBEDDINGS_PER_FEATURE = [1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194, \
                                           27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572]
BATCH_SIZE = 8
REDUCE_OPS = 'mean'

def check_equal(A, B):
    assert torch.allclose(A, B, rtol=1e-3, atol=1e-1)