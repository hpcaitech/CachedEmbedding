import torch
from torch import distributed as dist
from recsys.parallel import DDP as MyDDP
from torch.nn.parallel import DistributedDataParallel as TorchDDP

from functools import partial
import torch.multiprocessing as mp
import pytest

from colossalai.logging import disable_existing_loggers
from colossalai.testing import rerun_if_address_is_in_use
from colossalai.utils import free_port

from recsys import launch
from recsys.utils.distributed_manager import DISTMGR


class Net(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 3, bias=False)
        self.fc2 = torch.nn.Linear(3, 1, bias=False)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def _run_fwd_bwd(ddp_cls):
    if DISTMGR.get_world_size() > 1:
        return
    model = Net().cuda()
    w1 = model.fc1.weight
    w2 = model.fc2.weight
    # ddp_cls.set_params_to_ignore([w2])
    model = ddp_cls(model)
    x = torch.rand(2, 3, device=torch.cuda.current_device())
    logits = model(x)
    loss = torch.sum(logits)
    loss.backward()
    w1_grads = [torch.empty_like(w1) for _ in range(dist.get_world_size())]
    dist.all_gather(w1_grads, w1.grad)
    assert torch.equal(w1_grads[0], w1_grads[1])
    w2_grads = [torch.empty_like(w2) for _ in range(dist.get_world_size())]
    dist.all_gather(w2_grads, w2.grad)
    assert torch.equal(w2_grads[0], w2_grads[1])


def run(rank, world_size, port):
    disable_existing_loggers()
    launch(rank=rank, world_size=world_size, port=port, host='localhost', backend='nccl')
    assert DISTMGR.get_rank() == dist.get_rank()
    assert DISTMGR.get_world_size() == dist.get_world_size()

    _run_fwd_bwd(MyDDP)
    _run_fwd_bwd(TorchDDP)


@pytest.mark.parametrize('world_size', [2, 4])
@rerun_if_address_is_in_use()
def test_ddp(world_size):
    run_func = partial(run, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)


if __name__ == '__main__':
    test_ddp(4)