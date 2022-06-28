import os
import argparse

from .distributed_manager import DISTMGR


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, help='the master address for distributed training')
    parser.add_argument('--port', type=int, help='the master port for distributed training')
    parser.add_argument('--world_size', type=int, help='world size for distributed training')
    parser.add_argument('--rank', type=int, help='rank for the default process group')
    parser.add_argument('--local_rank', type=int, help='local rank on the node')
    parser.add_argument('--backend', type=str, default='nccl', help='backend for distributed communication')
    return parser


def launch(rank, world_size, host, port, backend, local_rank=None, seed=47):
    DISTMGR.init_default_process_group(rank, world_size, host, port, backend)
    DISTMGR.set_device(local_rank)

    DISTMGR.set_seed(seed)


def launch_from_torch(backend='nccl', seed=47):
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    host = os.environ['MASTER_ADDR']
    port = int(os.environ['MASTER_PORT'])
    launch(rank=rank, local_rank=local_rank, world_size=world_size, host=host, port=port, backend=backend, seed=seed)
