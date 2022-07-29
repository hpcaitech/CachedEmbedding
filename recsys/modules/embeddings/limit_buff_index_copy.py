import torch
from torch import LongTensor

class LimitBuffIndexCopyer(object):
    """LimitBuffIndexCopyer 
    Index Copy using limited temp buffer on CUDA.

    Args:
        size (int): buffer size
    """
    def __init__(self, size : int) -> None:
        pass

    def index_copy(self, dim : int, index : LongTensor, src : torch.Tensor, tgt : torch.Tensor):
        """copy 
        src tensor[index] -> tgt

        Args:
            dim (int): _description_
            index (int): _description_
            src (torch.Tensor): _description_
            tgt (torch.Tensor): _description_
        """
        # TODO(jiaruifang) split the copy into multiple times if index is too long.
        tgt.index_copy_(dim, index, src)