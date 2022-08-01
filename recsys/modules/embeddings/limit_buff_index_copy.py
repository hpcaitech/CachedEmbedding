import torch
from torch import LongTensor


class LimitBuffIndexCopyer(object):
    """LimitBuffIndexCopyer 
    Index Copy using limited temp buffer on CUDA.

    Args:
        size (int): buffer size
    """

    def __init__(self, size: int) -> None:
        self._buff_size = size

    @torch.no_grad()
    def index_copy(self, dim: int, src_index: LongTensor, tgt_index: LongTensor, src: torch.Tensor, tgt: torch.Tensor):
        """copy 
        src tensor[src_index] -(index_select)-> tmp -()-> tgt tensor [tgt_index]
        The valid part in src is continous, while in tgt is scatter.
        Args:
            dim (int):  dimension along which to index
            src_index (int): indices of src tensor to select from
            tgt_index (int): indices of tgt tensor to select from
            src (torch.Tensor):  the tensor containing values to copy
            tgt (torch.Tensor):  the tensor to be copied
        """
        # tgt.index_copy_(dim, index, src)
        assert dim == 0, "only support index_copy on dim 0"
        assert tgt.dim() == 2
        assert src.dim() == 2
        tgt_device = tgt.device
        src_device = src.device

        assert src_index.numel() == tgt_index.numel()
        dim_size = src_index.numel()
        src_index = src_index.to(src_device)
        for begin_pos in range(0, dim_size, self._buff_size):
            cur_len = min(self._buff_size, dim_size - begin_pos)
            src_idx_piece = src_index.narrow(0, begin_pos, cur_len)
            if src_device.type == 'cpu' and tgt_device.type == 'cuda':
                cpu_tmp_buffer = src.index_select(dim, src_idx_piece).pin_memory()
                tmp_buffer = torch.empty_like(cpu_tmp_buffer, device=tgt_device)
                tmp_buffer.copy_(cpu_tmp_buffer)
            else:
                tmp_buffer = src.index_select(dim, src_idx_piece).to(tgt_device)
            tgt_idx_piece = tgt_index.narrow(0, begin_pos, cur_len)
            tgt.index_copy_(dim, tgt_idx_piece, tmp_buffer)


if __name__ == '__main__':
    # TODO(jiaruifang) fix the unittest and move it test directory.
    src = torch.randn(10, 8)
    dst1 = torch.empty(123, 8)
    dst2 = torch.empty(123, 8)
    assert torch.allclose(dst1, dst2), f"{dst1-dst2}"
    idx = torch.tensor([3, 1, 29, 8, 5, 7, 12, 17, 33, 21])
    dst1.index_copy_(0, idx, src)

    copyer = LimitBuffIndexCopyer(3)
    copyer.index_copy(0, idx, src, dst2)
    assert torch.allclose(dst1, dst2), f"{dst1-dst2}"
