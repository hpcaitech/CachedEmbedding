#!/usr/bin/env python
# -*- encoding: utf-8 -*-

from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader


class BaseStreamDataIter(ABC):

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iter = iter(loader)
        self.stream = torch.cuda.Stream()
        self._preload()

    @staticmethod
    def _move_tensor(element):
        if torch.is_tensor(element):
            if not element.is_cuda:
                return element.cuda(non_blocking=True)
        return element

    def to_cuda(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cuda(non_blocking=True)
        elif isinstance(data, (list, tuple)):
            data_to_return = []
            for element in data:
                if isinstance(element, dict):
                    data_to_return.append({k: self._move_tensor(v) for k, v in element.items()})
                else:
                    data_to_return.append(self._move_tensor(element))
            data = data_to_return
        elif isinstance(data, dict):
            data = {k: self._move_tensor(v) for k, v in data.items()}
        else:
            raise TypeError(
                f"Expected batch data to be of type torch.Tensor, list, tuple, or dict, but got {type(data)}")
        return data
    
    @abstractmethod
    def _preload(self):
        pass

    @abstractmethod
    def _reset(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

