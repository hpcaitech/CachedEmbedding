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
from recsys.utils.extended_distributed import ext_all_to_all


def _collect_tensors(tensor, dim):
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()

    tensor = tensor.cpu()
    if rank == 0:
        gather_list = [torch.empty_like(tensor) for i in range(world_size)]
    else:
        gather_list = []

    torch.distributed.gather(tensor, gather_list, dst = 0, group = DISTMGR.get_cpu_group())
    if rank == 0:
        tensor_global = torch.cat(gather_list, dim=dim)
    else:
        tensor_global = None
    return tensor_global

def _test_all_to_all(use_cpu):
    device = torch.device('cpu') if use_cpu else torch.device('cuda', torch.cuda.current_device())
    rank = DISTMGR.get_rank()
    world_size = DISTMGR.get_world_size()

    B, H = 2 * world_size, 4 * world_size

    torch.manual_seed(0+rank)
    input_tensor = torch.randn(B, H//world_size)
    print(rank, input_tensor)
    # size (B//N, H)
    output_tensor = ext_all_to_all(input_tensor)

    input_global = _collect_tensors(input_tensor, 1)
    output_global = _collect_tensors(output_tensor, 0)
    if rank == 0:
        assert torch.allclose(input_global,output_global)


def run(rank, world_size, port, use_cpu):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    assert DISTMGR.get_rank() == dist.get_rank()
    assert DISTMGR.get_world_size() == dist.get_world_size()

    _test_all_to_all(use_cpu)


@pytest.mark.parametrize('world_size', [1, 4])
@pytest.mark.parametrize('use_cpu', [False, True])
@rerun_if_address_is_in_use()
def test_comm(world_size, use_cpu):
    run_func = partial(run, world_size=world_size, port=free_port(), use_cpu=use_cpu)
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_comm(4, False)
