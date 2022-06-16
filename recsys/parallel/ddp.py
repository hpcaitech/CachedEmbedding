import torch
import torch.distributed as dist
from recsys.utils.distributed_manager import DISTMGR

class DDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module
        self.comm_stream: torch.cuda.Stream = torch.cuda.Stream()

        # assume we only has dp for all processes
        self.dp_world_size = DISTMGR.get_world_size()
        for p in module.parameters():
            if p.requires_grad:
                p.register_hook(self.grad_handle)

    def parameters(self, recurse: bool = True):
        return self.module.parameters(recurse)

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        return self.module.named_parameters(prefix, recurse)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss: torch.Tensor):
        loss.backward()
        torch.cuda.current_stream().wait_stream(self.comm_stream)

    def grad_handle(self, grad):
        if grad.device.type != "cpu":
            if self.dp_world_size > 1:
                grad = grad / self.dp_world_size
                self.comm_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self.comm_stream):
                    group = DISTMGR.get_group()
                    dist.all_reduce(grad, group=group)

                grad.record_stream(self.comm_stream)
            return grad
        else:
            group = DISTMGR.get_cpu_group()
            dist.all_reduce(grad, group=group)
            return grad
