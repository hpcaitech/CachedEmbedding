import math
from typing import List, Callable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from recsys import DISTMGR, ParallelMode
from ..functional import reduce_forward

'''
Rebalancing hash function
'''

# Automatic load balancing, dynamic block_embedding_dims
class LoadBalanceManager(object):
    def __init__(self, field_dims: List[int], num_groups=4, base_emb_dim=128):
        assert len(field_dims) >= num_groups, \
                f"number of input fields {len(field_dims)} must be larger than the world size {num_groups}"
        self.field_dims = field_dims
        self.num_groups = num_groups
        self.base_emb_dim = base_emb_dim
        self._initialize()

    def _initialize(self):
        dim_indices = np.array(range(len(self.field_dims)))
        np.random.shuffle(dim_indices)
        chunk_size = len(self.field_dims) // self.num_groups
        self.groups = []

        for i in range(self.num_groups):
            if i == self.num_groups-1:
                self.groups.append(dim_indices[i*chunk_size:])
                break
            self.groups.append(dim_indices[i*chunk_size:(i+1)*chunk_size])

        self.emb_dims = []
        total_sum = sum(self.field_dims)
        for group in self.groups:
            div = total_sum / sum([self.field_dims[x] for x in group])
            # scale base embedding dim by total_sum/sum
            emb_dim = int(self.base_emb_dim / 2**(int(math.log2(div))))
            self.emb_dims.append(emb_dim)

    def mapping_rule(self, _input: torch.Tensor, rank: int) -> torch.Tensor:
        group = self.groups[rank]
        assert min(group) >= 0 and max(group) < _input.size(1)
        return _input[:, group]
    

def balance_hash(field_dims: List[int], num_groups=4, base_emb_dim=128) \
                                            -> Tuple[List[int], Callable]:
    assert len(field_dims) >= num_groups, \
            f"number of input fields {len(field_dims)} must be larger than the world size {num_groups}"
    dim_indices = np.array(range(len(field_dims)))
    np.random.shuffle(dim_indices)
    chunk_size = len(field_dims) // num_groups
    groups = []

    for i in range(num_groups):
        if i == num_groups-1:
            groups.append(dim_indices[i*chunk_size:])
            break
        groups.append(dim_indices[i*chunk_size:(i+1)*chunk_size])

    emb_dims = []
    total_sum = sum(field_dims)
    for group in groups:
        div = total_sum / sum([field_dims[x] for x in group])
        # scale base embedding dim by total_sum/sum
        emb_dim = int(base_emb_dim / 2**(int(math.log2(div))))
        emb_dims.append(emb_dim)

    def mapping_rule(groups: List[List[int]], _input: torch.Tensor, rank: int) -> torch.Tensor:
        group = groups[rank]
        assert min(group) >= 0 and max(group) < _input.size(1)
        return _input[:, group]

    return groups, emb_dims, mapping_rule


class LambdaLayer(nn.Module):

    def __init__(self, func: Callable):
        super(LambdaLayer, self).__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)


class BlockEmbeddingBag(nn.Module):

    def __init__(self,
                 num_embeddings: int,
                 block_embedding_dim: int = 64,
                 base_embedding_dim: int = 128,
                 padding_idx=None,
                 max_norm=None,
                 norm_type=2.,
                 scale_grad_by_freq=False,
                 sparse=False,
                 mode='mean',
                 include_last_offset=False,
                 embed_w=None,
                 linear_w=None,
                 freeze_w=False,
                 device=None,
                 dtype=None,
                 init_method=torch.nn.init.xavier_normal_):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.block_embedding_dim = block_embedding_dim
        self.base_embedding_dim = base_embedding_dim
        
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        # Specific to embedding bag
        self.mode = mode
        self.include_last_offset = include_last_offset
        
        if embed_w is None:
            self.embed_w = torch.nn.Parameter(
                torch.empty(num_embeddings, block_embedding_dim, device=device, dtype=dtype))

            init_method(self.embed_w)
            if padding_idx is not None:
                with torch.no_grad():
                    self.weight[padding_idx].fill_(0)
        else:
            assert list(embed_w.shape) == [num_embeddings, block_embedding_dim]
            self.embed_weight = torch.nn.Parameter(embed_w)

        self.embed_weight.requires_grad = not freeze_w

        if block_embedding_dim == base_embedding_dim:
            self.projector = LambdaLayer(lambda x:x)
        else:
            self.projector = nn.Linear(
                                in_features=block_embedding_dim,
                                out_features=base_embedding_dim,
                                device=device
                            )
            if linear_w is not None:
                init_method(self.projector.weight)
            else:
                assert list(linear_w.shape) == [block_embedding_dim, base_embedding_dim]
                with torch.no_grad():
                    self.projector.weight = nn.Parameter(linear_w)
        
        self.projector.weight.requires_grad = not freeze_w

    def forward(self, input_, offsets=None, per_sample_weights=None):
        output_parallel = F.embedding_bag(input_, self.embed_weight, offsets, self.max_norm, self.norm_type,
                                          self.scale_grad_by_freq, self.mode, self.sparse, per_sample_weights,
                                          self.include_last_offset, self.padding_idx)

        mapped_output_parallel = self.projector(output_parallel)
        assert mapped_output_parallel.size() == (input_.size(0), self.base_embedding_dim)
        return mapped_output_parallel

    @classmethod
    def from_pretrained(cls,
                        weights: List[torch.Tensor],
                        base_embedding_dim: int = 128,
                        freeze: bool = True,
                        max_norm: Optional[float] = None,
                        norm_type: float = 2.,
                        scale_grad_by_freq: bool = False,
                        mode: str = 'mean',
                        sparse: bool = False,
                        include_last_offset: bool = False,
                        padding_idx: Optional[int] = None,
                        parallel_mode=None,
                        init_method=torch.nn.init.xavier_normal_) -> 'BlockEmbeddingBag':
        assert len(weights) == 2 and weights[0].dim() == 2, \
            'Both embedding and linear weights are expected; embeddings parameter is expected to be 2-dimensional'
        rows, cols = weights[0].shape
        embeddingbag = cls(num_embeddings=rows,
                           block_embedding_dim=cols,
                           base_embedding_dim=base_embedding_dim,
                           embed_w=weights[0],
                           linear_w=weights[1],
                           freeze_w=freeze,
                           max_norm=max_norm,
                           norm_type=norm_type,
                           scale_grad_by_freq=scale_grad_by_freq,
                           mode=mode,
                           sparse=sparse,
                           include_last_offset=include_last_offset,
                           padding_idx=padding_idx,
                           parallel_mode=parallel_mode,
                           init_method=init_method)
        return embeddingbag


class ParallelMixVocabEmbeddingBag(nn.Module):

    def __init__(self,
                field_dims: List[int], # change of interface (not sum of field_dims)
                embedding_dim: int = 128,
                parallel_mode = None,
                *args,
                **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Decide number of nodes
        self.parallel_mode = ParallelMode.DEFAULT if parallel_mode is None else parallel_mode
        self.world_size = DISTMGR.get_world_size(self.parallel_mode)
        self.rank = DISTMGR.get_rank(self.parallel_mode)

        self.load_balance_mgr = LoadBalanceManager(field_dims, self.world_size)
        # self.vocab_groups, self.block_emb_dims, self.mapping_rule = balance_hash(field_dims, self.world_size)

        group = self.load_balance_mgr.groups[self.rank]
        block_dim = self.load_balance_mgr.emb_dims[self.rank]
        self.comm_func = reduce_forward # need all_reduce

        self.embed = BlockEmbeddingBag(
                                sum([field_dims[i] for i in group]), 
                                block_embedding_dim=block_dim,
                                base_embedding_dim=self.embedding_dim,
                                *args,
                                **kwargs)

    def forward(self, x, offsets=None):
        x_parallel = self.load_balance_mgr.mapping_rule(x, self.rank)
        output_parallel = self.embed(x_parallel, offsets)
        output_gather = self.comm_func(output_parallel, self.parallel_mode)

        assert output_gather.shape == (x.size(0), self.embedding_dim)
        return output_gather
