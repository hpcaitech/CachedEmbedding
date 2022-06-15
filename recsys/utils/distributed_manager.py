from enum import Enum
import random
import numpy as np
import torch
from recsys.utils.singleton_meta import SingletonMeta


class ParallelMode(Enum):
    DEFAULT = 'default'

    DATA = "data_parallel"
    TENSOR_PARALLEL = "tensor_parallel"


class DistributedManager(metaclass=SingletonMeta):

    def __init__(self):
        self.process_groups = dict()
        self.cpu_process_groups = dict()
        self.ranks_in_group = dict()

    def init_default_process_group(self, rank, world_size, host, port, backend):
        init_method = f"tcp://{host}:{port}"
        torch.distributed.init_process_group(rank=rank, world_size=world_size, backend=backend, init_method=init_method)
        group = torch.distributed.GroupMember.WORLD

        ranks = list(range(world_size))
        cpu_group = torch.distributed.new_group(ranks, backend='gloo') \
            if backend != 'gloo' else torch.distributed.GroupMember.WORLD

        self.add_process_group(ParallelMode.DEFAULT, group, cpu_group, ranks)

    def add_process_group(self, name, group, cpu_group, ranks):
        assert name not in self.process_groups, \
            f"Process group name: {name} is already in use"
        self.process_groups[name] = group
        self.cpu_process_groups[name] = cpu_group
        self.ranks_in_group[name] = ranks

    def new_process_group(self, this_parallel_size, mode):
        rank = self.get_rank()
        world_size = self.get_world_size()

        process_group = None
        cpu_group = None
        ranks_in_group = None

        num_group = world_size // this_parallel_size

        for i in range(num_group):
            ranks = [i + j * num_group for j in range(this_parallel_size)]
            group = torch.distributed.new_group(ranks)
            group_cpu = torch.distributed.new_group(ranks, backend='gloo') \
                if torch.distributed.get_backend() != 'gloo' else group

            if rank in ranks:
                process_group = group
                cpu_group = group_cpu
                ranks_in_group = ranks
        self.add_process_group(mode, process_group, cpu_group, ranks_in_group)

    def get_distributed_info(self):
        info_str = "\n"
        for key in self.process_groups:
            info_str += f">>> Mode: {key}, Rank: {self.get_rank(key)}, world size: {self.get_world_size(key)}" + "\n"
        return info_str

    def get_world_size(self, name=ParallelMode.DEFAULT):
        return torch.distributed.get_world_size(self.process_groups[name])

    def get_rank(self, name=ParallelMode.DEFAULT):
        return torch.distributed.get_rank(self.process_groups[name])

    def get_group(self, name=ParallelMode.DEFAULT):
        return self.process_groups[name]

    def get_cpu_group(self, name=ParallelMode.DEFAULT):
        return self.cpu_process_groups[name]

    def get_ranks_in_group(self, name=ParallelMode.DEFAULT):
        return self.ranks_in_group[name]

    def set_device(self, device_ordinal=None):
        if torch.cuda.is_available():
            global_rank = self.get_rank()
            if device_ordinal is None:
                device_ordinal = global_rank % torch.cuda.device_count()
            torch.cuda.set_device(device_ordinal)

    def set_seed(self, seed):
        """
        To achieve reproducible results, it's necessary to fix random seeds
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


DISTMGR = DistributedManager()
