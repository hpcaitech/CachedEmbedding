import torch
import torch.distributed as dist

from colossalai.utils.cuda import get_current_device


class _CopyInputToCPU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        # logger.debug(f"Copy input to cpu and {input_.dtype}.")
        return input_.to(torch.device("cpu"))

    @staticmethod
    def backward(ctx, grad_output):
        target_device = torch.device(get_current_device())
        # logger.debug("Copy grad_output to cuda.")
        return grad_output.to(target_device)


class _CopyActToGPU(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        return input_.to(get_current_device())

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.to(torch.device("cpu"))


def copy_to_cpu(input_):
    return _CopyInputToCPU.apply(input_)


def copy_to_gpu(input_):
    return _CopyActToGPU.apply(input_)
