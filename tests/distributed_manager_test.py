import pytest
from functools import partial
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port

from recsys import launch
from recsys import DISTMGR

SIZE = 8


def all_reduce(use_cpu):
    device = torch.device('cpu') if use_cpu else torch.device('cuda', torch.cuda.current_device())
    local_tensor = torch.tensor([DISTMGR.get_rank() * SIZE + j for j in range(SIZE)]).to(device)
    buffer = local_tensor.clone()
    print(f"Rank: {DISTMGR.get_rank()}, before: {local_tensor}")

    dist.all_reduce(buffer, group=(DISTMGR.get_cpu_group() if use_cpu else DISTMGR.get_group()))
    print(f"Rank: {DISTMGR.get_rank()}, after: {buffer}")


def run(rank, world_size, port, use_cpu):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    assert DISTMGR.get_rank() == dist.get_rank()
    assert DISTMGR.get_world_size() == dist.get_world_size()
    print(f"Rank: {DISTMGR.get_rank()}, world size: {DISTMGR.get_world_size()}")

    all_reduce(use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_comm(world_size, use_cpu):
    run_func = partial(run, world_size=world_size, port=free_port(), use_cpu=use_cpu)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_comm(4, False)
