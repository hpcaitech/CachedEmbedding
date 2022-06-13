import os

from .distributed_manager import distributed_manager


def launch(rank, world_size, host, port, backend, local_rank=None, seed=47):
    distributed_manager.init_default_process_group(rank, world_size, host, port, backend)
    distributed_manager.set_device(local_rank)

    distributed_manager.set_seed(seed)


def launch_from_torch(backend='nccl'):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    launch(rank=rank,
           local_rank=local_rank,
           world_size=world_size,
           host=host,
           port=port,
           backend=backend)
