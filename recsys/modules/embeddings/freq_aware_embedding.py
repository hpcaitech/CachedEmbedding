import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Iterator, Tuple

from .base_embeddings import BaseEmbeddingBag
from .chunk_param_mgr import ChunkParamMgr
from torch.nn.parameter import Parameter


class FreqAwareEmbeddingBag(BaseEmbeddingBag):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, *args, **kwargs):
        super(FreqAwareEmbeddingBag, self).__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self._weight = torch.randn(self.num_embeddings, self.embedding_dim, device='cpu', dtype=dtype)

    def preprocess(self, chunk_size: int, cuda_chunk_num: int, ids_freq_mapping: Optional[List[int]] = None, use_warmup = True):
        """
        Called after initialized. 
        Reorder the weight rows according to the ids_freq_mapping.
        Then, let the weights of the Module be managed by a ChunkParamMgr.
        Args:
            chunk_size (int): chunk size
            cuda_chunk_num (int): number of chunk can be hosted in CUDA memory
            ids_freq_mapping (List[int]): a list, idx is id number, value is freq
        """
        self.chunk_weight_mgr = ChunkParamMgr(self._weight, chunk_size, cuda_chunk_num)
        self.chunk_weight_mgr.reorder(ids_freq_mapping, use_warmup)

    def forward(self, indices, offsets=None, per_sample_weights=None):
        with torch.no_grad():
            reorder_ids = self.chunk_weight_mgr.prepare_ids(indices)

        embeddings = F.embedding_bag(reorder_ids, self.chunk_weight_mgr.cuda_partial_weight, offsets, self.max_norm,
                                     self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                     per_sample_weights, self.include_last_offset, self.padding_idx)

        return embeddings

    @property
    def weight(self):
        assert self.chunk_weight_mgr is not None
        return self.chunk_weight_mgr.cpu_weight.narrow(0, 0, self.num_embeddings)

    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[Tuple[str, Parameter]]:
        yield 'weight', self.chunk_weight_mgr.cuda_partial_weight

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield self.chunk_weight_mgr.cuda_partial_weight

    @property
    def num_hits_history(self):
        return self.chunk_weight_mgr.num_hits_history

    @property
    def num_miss_history(self):
        return self.chunk_weight_mgr.num_miss_history

    @property
    def num_write_back_history(self):
        return self.chunk_weight_mgr.num_write_back_history

    @property
    def swap_in_bandwidth(self):
        if self.chunk_weight_mgr._cpu_to_cuda_numel > 0:
            return self.chunk_weight_mgr._cpu_to_cuda_numel * self.chunk_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.chunk_weight_mgr._cpu_to_cuda_elpase
        else:
            return 0

    @property
    def swap_out_bandwidth(self):
        if self.chunk_weight_mgr._cuda_to_cpu_numel > 0:
            return self.chunk_weight_mgr._cuda_to_cpu_numel * self.chunk_weight_mgr.elem_size_in_byte / 1e6 / \
                   self.chunk_weight_mgr._cuda_to_cpu_elapse
        return 0

    @property
    def input_id_percent_in_load_chunk(self):
        return 0    # np.mean(self.chunk_weight_mgr.input_id_percent_in_load_chunk) * 100
