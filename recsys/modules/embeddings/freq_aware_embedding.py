import torch
import torch.nn.functional as F
from typing import List, Optional

from .base_embeddings import BaseEmbeddingBag
from .chunk_param_mgr import ChunkParamMgr

class FreqAwareEmbeddingBag(BaseEmbeddingBag):

    def __init__(self, num_embeddings, embedding_dim, dtype=None, *args, **kwargs):
        super(FreqAwareEmbeddingBag, self).__init__(num_embeddings, embedding_dim, *args, **kwargs)
        self._weight = torch.randn(self.num_embeddings, self.embedding_dim, device='cpu', dtype=dtype)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _preprocess(self, chunk_size: int, cuda_chunk_num: int, ids_freq_mapping: Optional[List[int]] = None):
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
        self.chunk_weight_mgr.reorder(ids_freq_mapping)

    def forward(self, indices, offsets=None, per_sample_weights=None):
        reorder_ids = self.chunk_weight_mgr.prepare_ids(indices)

        embeddings = F.embedding_bag(reorder_ids, self.chunk_weight_mgr.cuda_partial_weight, offsets, self.max_norm,
                                     self.norm_type, self.scale_grad_by_freq, self.mode, self.sparse,
                                     per_sample_weights, self.include_last_offset, self.padding_idx)

        return embeddings

    @property
    def weight(self):
        return self.chunk_weight_mgr.cpu_weight.narrow(0, 0, self.num_embeddings)
