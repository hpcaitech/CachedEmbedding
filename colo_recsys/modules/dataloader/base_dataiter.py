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

