#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from typing import TypeVar, Iterator

import torch
from torch.utils.data import Sampler, Dataset, DataLoader

from .base_dataiter import BaseStreamDataIter


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
