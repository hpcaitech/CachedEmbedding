import torch

from .. import DISTMGR as dist_manager


def _reduce(x, parallel_mode):
    if dist_manager.get_world_size(parallel_mode) == 1:
        return x

    process_group = dist_manager.get_cpu_group(parallel_mode) \
        if x.device.type == 'cpu' else dist_manager.get_group(parallel_mode)
    torch.distributed.all_reduce(x, group=process_group)
    return x


def _gather(x, parallel_mode, dim):
    world_size = dist_manager.get_world_size(parallel_mode)
    if world_size == 1:
        return x

    rank = dist_manager.get_rank(parallel_mode)
    process_group = dist_manager.get_cpu_group(parallel_mode) \
        if x.device.type == 'cpu' else dist_manager.get_group(parallel_mode)

    tensor_list = [torch.empty_like(x) if i != rank else x for i in range(world_size)]
    torch.distributed.all_gather(tensor_list, x, group=process_group)
    return torch.cat(tensor_list, dim=dim).contiguous()


def _tensor_gather(x, parallel_mode, dim):
    world_size = dist_manager.get_world_size(parallel_mode)
    if world_size == 1:
        return x

    rank = dist_manager.get_rank(parallel_mode)
    process_group = dist_manager.get_cpu_group(parallel_mode) \
        if x.device.type == 'cpu' else dist_manager.get_group(parallel_mode)

    tensor_list = [None if i != rank else x for i in range(world_size)]
    torch.distributed.all_gather_object(tensor_list, x, group=process_group)
    result = torch.cat([each.to(x.device) for each in tensor_list], dim=dim).contiguous()

    return result


def _tensor_split(x, parallel_mode, dim):
    world_size = dist_manager.get_world_size(parallel_mode)
    if world_size == 1:
        return x

    rank = dist_manager.get_rank(parallel_mode)
    tensor = torch.tensor_split(x, world_size, dim=dim)[rank]
    return tensor


class _ReduceForward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, parallel_mode):
        return _reduce(x, parallel_mode)

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def reduce_forward(x, parallel_mode):
    return _ReduceForward.apply(x, parallel_mode)


class _TensorGatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, parallel_mode, dim):
        ctx.parallel_mode = parallel_mode
        ctx.dim = dim
        return _tensor_gather(x, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad):
        return _tensor_split(grad, ctx.parallel_mode, ctx.dim), None, None


def tensor_gather_forward_split_backward(x, parallel_mode, dim):
    return _TensorGatherForwardSplitBackward.apply(x, parallel_mode, dim)


class _GatherForwardSplitBackward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, parallel_mode, dim):
        ctx.parallel_mode = parallel_mode
        ctx.dim = dim
        return _gather(x, parallel_mode, dim)

    @staticmethod
    def backward(ctx, grad):
        return _tensor_split(grad, ctx.parallel_mode, ctx.dim), None, None


def gather_forward_split_backward(x, parallel_mode, dim):
    return _GatherForwardSplitBackward.apply(x, parallel_mode, dim)
