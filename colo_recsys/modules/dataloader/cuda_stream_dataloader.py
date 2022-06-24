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
            self.next_input, self.next_target = next(self.iter)
       
        except StopIteration:
            self.next_input = None
            self.next_target = None
            self._reset()
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)

    def _reset(self):
        self.iter = iter(self.loader)
        self.stream = torch.cuda.Stream()

        self._preload()

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        
        self._preload()
        return input, target

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
