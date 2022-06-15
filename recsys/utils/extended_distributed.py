import torch
from .distributed_manager import DISTMGR

def ext_all_to_all(input_tensor) -> torch.Tensor:
    """
    input tensor (B, H/world_size)
    output tensor (B/world_size, H)
    """
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()

    
    # move cpu tensor to gpu
    if input_tensor.device.type == 'cpu':
        input_tensor = input_tensor.to(rank)

    assert input_tensor.dim() == 2, f"input tensor must has 2 dimensions"

    B, H_div_N = input_tensor.size()
    H = H_div_N * world_size
    dtype = input_tensor.dtype

    input_tensor_list = list(torch.split(input_tensor, B//world_size, 0))
    output_tensor_list = [torch.empty(B//world_size, H//world_size, dtype = dtype).cuda(rank) for i in range(world_size)]

    # nccl all2all
    torch.distributed.all_to_all(output_tensor_list, input_tensor_list)

    out_tensor = torch.cat(output_tensor_list, dim=1)
    assert out_tensor.size()[0] == B/world_size and out_tensor.size()[1] == H


    return out_tensor