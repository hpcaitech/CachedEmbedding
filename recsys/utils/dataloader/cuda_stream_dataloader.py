#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import math
from typing import TypeVar, Iterator
import random

import torch
from torch.utils.data import Sampler, Dataset, DataLoader
import numpy as np

from ..distributed_manager import ParallelMode, DISTMGR
from .base_dataiter import BaseStreamDataIter

T_co = TypeVar('T_co', covariant=True)


class DataParallelSampler(Sampler):
    """A data sampler for distributed data parallelism.
    Args:
        dataset (:class:`torch.utils.data.Dataset`): The Dataset for sampling.
        shuffle (bool, optional): Whether to shuffle data, defaults to False.
        seed (int, optional): The random seed used for sampling, defaults to 0.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
    """

    def __init__(self, dataset: Dataset, shuffle: bool = False, seed: int = 0, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.num_replicas = DISTMGR.get_world_size(ParallelMode.DATA)
        self.rank = DISTMGR.get_rank(ParallelMode.DATA)
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        # type: ignore[arg-type]
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / \
                self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)    # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # type: ignore[arg-type]
            indices = torch.randperm(len(self.dataset), generator=g).tolist()

            # update for next epoch so that there is no need to call
            # set_epoch manually
            self.epoch += 1
        else:
            indices = list(range(len(self.dataset)))    # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.
        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


def get_dataloader(dataset,
                   shuffle=False,
                   seed=1024,
                   add_sampler=True,
                   drop_last=False,
                   pin_memory=False,
                   num_workers=0,
                   **kwargs):
    r"""Set up a deterministic dataloader (also configure seed workers, samplers and whether shuffle or not)
    Note:
        When pipeline parallel is enabled, shuffle cannot be True as it will result in mismatch between input data
        on the 1st stage and label on the last stage.
    Args:
        dataset (:class:`torch.utils.data.Dataset`): The dataset to be loaded.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
        seed (int, optional): Random worker seed for sampling, defaults to 1024.
        add_sampler: Whether to add ``DistributedDataParallelSampler`` to the dataset. Defaults to True.
        drop_last (bool, optional): Set to True to drop the last incomplete batch, if the dataset size
            is not divisible by the batch size. If False and the size of dataset is not divisible by
            the batch size, then the last batch will be smaller, defaults to False.
        pin_memory (bool, optional): Whether to pin memory address in CPU memory. Defaults to False.
        num_workers (int, optional): Number of worker threads for this dataloader. Defaults to 0.
        kwargs (dict): optional parameters for ``torch.utils.data.DataLoader``, more details could be found in
                `DataLoader <https://pytorch.org/docs/stable/_modules/torch/utils/data/dataloader.html#DataLoader>`_.
    Returns:
        :class:`torch.utils.data.DataLoader`: A DataLoader used for training or testing.
    """
    _kwargs = kwargs.copy()

    if add_sampler and DISTMGR.is_initialized(ParallelMode.DATA) and DISTMGR.get_world_size(ParallelMode.DATA) > 1:
        sampler = DataParallelSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    # Deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    if sampler is None:
        return DataLoader(dataset,
                          worker_init_fn=seed_worker,
                          shuffle=shuffle,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)
    else:
        return DataLoader(dataset,
                          sampler=sampler,
                          worker_init_fn=seed_worker,
                          drop_last=drop_last,
                          pin_memory=pin_memory,
                          num_workers=num_workers,
                          **_kwargs)


class CudaStreamDataIter(BaseStreamDataIter):
    """
    A data iterator that supports batch prefetching with the help of cuda stream. 
    """

    def __init__(self, loader: DataLoader):
        super().__init__(loader)

    def _preload(self):
        try:
            self.batch_data = next(self.iter)

        except StopIteration:
            self.batch_data = None
            self._reset()
            return

        with torch.cuda.stream(self.stream):
            self.batch_data = self.to_cuda(self.batch_data)

    def _reset(self):
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self._preload()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data

        if batch_data is not None:
            self.record_stream(batch_data, torch.cuda.current_stream())

        self._preload()
        return batch_data

    def __iter__(self):
        return self


class FiniteDataIter(BaseStreamDataIter):

    def _reset(self):
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()
        self._preload()

    def __init__(self, data_loader):
        super(FiniteDataIter, self).__init__(data_loader)

    def _preload(self):
        try:
            self.batch_data = next(self.iter)

            with torch.cuda.stream(self.stream):
                self.batch_data = self.batch_data.to(torch.cuda.current_device(), non_blocking=True)

        except StopIteration:
            self.batch_data = None

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        if batch_data is not None:
            batch_data.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration()

        self._preload()
        return batch_data

    def __iter__(self):
        return self


class CudaStreamDataloader(object):

    def __init__(self, init_dataloader: DataLoader):
        self.loader = init_dataloader

    def __iter__(self) -> CudaStreamDataIter:
        return CudaStreamDataIter(self.loader)

    def __len__(self):
        return len(self.loader)


def get_cuda_stream_dataloader(dataset,
                               shuffle=False,
                               seed=1024,
                               add_sampler=True,
                               drop_last=False,
                               pin_memory=False,
                               num_workers=0,
                               **kwargs):

    dataloader = get_dataloader(dataset,
                                shuffle=shuffle,
                                seed=seed,
                                add_sampler=add_sampler,
                                drop_last=drop_last,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                **kwargs)

    return CudaStreamDataloader(dataloader)
