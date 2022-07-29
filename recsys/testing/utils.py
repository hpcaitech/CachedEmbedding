import numpy as np
import torch
from .. import DISTMGR


def synthesize_1d_sparse_feature(
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


def gather_tensor(tensor):
    gather_list = []
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]

    group = DISTMGR.get_group()
    torch.distributed.gather(tensor, gather_list, dst=0, group=group)
    return gather_list
