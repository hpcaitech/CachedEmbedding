#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import torch
from torch.utils.data import DataLoader
from colossalai.utils import get_dataloader

from .base_dataiter import BaseStreamDataIter


class CudaStreamDataIter(BaseStreamDataIter):
    """
    A data iterator that supports batch prefetching with the help of cuda stream. 
    Be aware that it now only supports batch loading on GPU.
    Also, it can only support dataset in the format of (input, target/label) 
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
            batch_data.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return batch_data

    def __iter__(self):
        return self


class CudaStreamDataloader(object):

    def __init__(self,
                 init_dataloader: DataLoader):
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
